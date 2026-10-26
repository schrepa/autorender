//! Screen effects: ambient occlusion and bloom, the haze
//! and screen-space reflections.
//!
//! Frames with an effect draw the scene into a color target whose alpha is the
//! ambient share of each pixel's light, and a glow target of each pixel's
//! exposed brightness above the bloom threshold and its reflectivity. Ambient
//! occlusion samples the depth buffer at half resolution, and one
//! half-resolution filter smooths it on each surface. The haze marches the
//! near air through the shadow clipmap at half resolution, and its filter
//! blends that over the last displayed frame's result, reprojected, which a
//! copy keeps for the next. Reflections walk a pyramid of the depth buffer's
//! nearest depths at half resolution and filter the result; bloom filters the
//! glowing light down a chain of half-resolution levels and back up.
//! Temporally antialiased frames apply them in their resolve, other frames in
//! a composite that writes the canvas, upscaling the scene to the display.
//!
//! Targets are made at the display's size, or a fraction of it; frames draw
//! their render size of them from the top-left corner, every pass within
//! its share, and read within it.

#![expect(
    clippy::expect_used,
    reason = "encoding reads the targets the frame's preparation sized"
)]

use crate::shaders::entry::{haze, post};
use engine_core::{Applied, RenderSnapshot, glam::Mat4};

use crate::bindings::{HAZE, POST};
use crate::cached::{Bound, GroupCache, Versioned};
use crate::depth_pyramid::{DepthPyramid, Reduction};
use crate::gpu_data::{HAZE_SHADER, POST_SHADER, PostUniform};
use crate::pipelines::{self, Fillers, Fullscreen, filled_group, fullscreen};
use crate::profiling::{GpuProfiler, Span};
use crate::render_features::{Encode, FeaturePass, RenderFeature};
use crate::render_graph::{Access, BLACK, Clear, FrameShape, Node, Pass, RenderGraph, Resource};
use crate::screen_effects::{Effects, UniformFrame};
use crate::targets::{FrameRegion, FrameSize, ScreenTarget, draw_within};

/// Linear color with 8 perceptual bits; alpha holds the ambient share.
pub(crate) const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Glow weight and reflectivity.
pub(crate) const GLOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;
/// The surface reflections follow: the shading normal,
/// octahedral, and roughness, 10 bits each.
pub(crate) const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgb10a2Unorm;
/// The filtered occlusion's visibility and each texel's inverse distance,
/// which the composite and resolve read.
const OCCLUSION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Occlusion before its filter, and each texel's distance from the eye.
const RAW_OCCLUSION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Bloom levels: light beyond 1 without alpha, packed in 32 bits where the
/// device renders them (`rg11b10ufloat-renderable`).
fn bloom_format(device: &wgpu::Device) -> wgpu::TextureFormat {
    if device
        .features()
        .contains(wgpu::Features::RG11B10UFLOAT_RENDERABLE)
    {
        wgpu::TextureFormat::Rg11b10Ufloat
    } else {
        wgpu::TextureFormat::Rgba16Float
    }
}
/// The scene's blurred levels: averages of the 8-bit sRGB scene.
const SCENE_LEVEL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Reflected color, from the scene and its levels, and weight.
const REFLECTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// The haze's light, below zero where its drifting air is clearer than its
/// mean or sees less of the sky than `atmospheric` counts, and each texel's
/// inverse distance: its march, its result filtered over time and that
/// result's copy the next frame reprojects, at half resolution.
const HAZE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Half-resolution bloom levels; each halves the one before.
const BLOOM_LEVELS: u32 = 6;
/// Blurred half-resolution levels of the scene rough reflections read.
const SCENE_LEVELS: u32 = 6;

/// Where the haze's result filtered over time and the copy of it the next
/// frame's filter reprojects lie among the targets' textures.
const HAZE_TEXTURE: usize = 10;
const HAZE_HISTORY: usize = 11;

struct Targets {
    // Scene, glow, occlusion, bloom, reflections, the haze's march, traced
    // reflections, raw occlusion, surface, scene levels, and the haze
    // filtered and its history, destroyed with the targets.
    textures: [wgpu::Texture; 12],
    scene: wgpu::TextureView,
    glow: wgpu::TextureView,
    surface: wgpu::TextureView,
    // Every blurred scene level, which reflections read, and group `i`
    // reading the scene (0) or level `i - 1` to write level `i`.
    scene_levels: wgpu::TextureView,
    scene_level_views: Vec<wgpu::TextureView>,
    scene_level_groups: Vec<wgpu::BindGroup>,
    // Filtered occlusion, which the composite and resolve read, and the raw
    // occlusion the filter reads.
    occlusion: wgpu::TextureView,
    raw_occlusion: wgpu::TextureView,
    // Filtered reflections, which the composite and resolve read, and the
    // traced ones the filter reads.
    reflections: wgpu::TextureView,
    traced: wgpu::TextureView,
    // The haze: this frame's march, which its filter blends over the
    // history; the filtered result, which the composite and resolve read and
    // the history copies for the next frame.
    marched: wgpu::TextureView,
    haze: wgpu::TextureView,
    history: wgpu::TextureView,
    // One view per bloom level, and the size of the half-resolution chains'
    // first levels, which each level halves.
    bloom: Vec<wgpu::TextureView>,
    levels: (u32, u32),
    bloom_first: wgpu::BindGroup,
    // Group `i` reads level `i` (down, writing `i + 1`) or level `i + 1`
    // (up, adding into `i`).
    bloom_down: Vec<wgpu::BindGroup>,
    bloom_up: Vec<wgpu::BindGroup>,
    // Frames that reflect and bloom halve both chains in the same passes:
    // the first from the scene, group `i` from scene and bloom level `i`.
    levels_first: wgpu::BindGroup,
    levels_down: Vec<wgpu::BindGroup>,
    composite: wgpu::BindGroup,
}

impl Drop for Targets {
    fn drop(&mut self) {
        for texture in &self.textures {
            texture.destroy();
        }
    }
}

pub(crate) struct PostPass {
    // The layout of every post.wgsl pass and fillers for the bindings each
    // leaves out.
    layout: crate::pipelines::GroupLayout,
    fillers: Fillers,
    occlusion: wgpu::RenderPipeline,
    // Filters occlusion on each surface.
    effects_filter: wgpu::RenderPipeline,
    reflections: wgpu::RenderPipeline,
    reflections_filter: wgpu::RenderPipeline,
    // Nearest depths the reflection rays walk, created with the first frame
    // that reflects.
    nearest: DepthPyramid,
    // The pixel of each 2x2 block this frame's reflections trace.
    trace_index: u32,
    // The haze's march, which reads the camera group of the main passes
    // (shadow maps, lighting and terrains' bounce probes) and its own, and
    // its filter over time.
    haze: wgpu::RenderPipeline,
    haze_layout: crate::pipelines::GroupLayout,
    haze_filter: wgpu::RenderPipeline,
    // What the last submitted display frame left the haze's filter (none
    // after any other frame), what this frame leaves once it is submitted,
    // and the display frames that left one, which turn the march's jitter.
    haze_history: Option<HazeHistory>,
    haze_kept: Option<HazeHistory>,
    haze_frames: u32,
    bloom_first: wgpu::RenderPipeline,
    bloom_down: wgpu::RenderPipeline,
    bloom_up: wgpu::RenderPipeline,
    // The scene levels bloom leaves, halved as bloom halves its own.
    scene_down: wgpu::RenderPipeline,
    bloom_format: wgpu::TextureFormat,
    // Both chains' levels at once, into two targets.
    levels_first: wgpu::RenderPipeline,
    levels_down: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    written: Option<PostUniform>,
    sampler: wgpu::Sampler,
    targets: ScreenTarget<Targets>,
    // The frame's sizes: the targets', the display's, and the part of them
    // it draws.
    region: FrameRegion,
    // Occlusion, filter, reflection and haze bindings over the depth buffer
    // and targets (and reflections', the pyramid).
    occlusion_group: GroupCache<5>,
    effects_filter_group: GroupCache<3>,
    reflections_group: GroupCache<8>,
    filter_group: GroupCache<3>,
    haze_group: GroupCache<2>,
    haze_filter_group: GroupCache<4>,
}

/// What a display frame's haze leaves the next frame's filter: its
/// unjittered view-projection and eye, which the next frame reprojects the
/// filtered haze from, and the half-resolution texels of it it drew.
#[derive(Clone, Copy)]
struct HazeHistory {
    view_projection: Mat4,
    eye: engine_core::glam::Vec3,
    drawn: (u32, u32),
}

impl PostPass {
    /// `camera_layout` is the first group of the main passes, which the
    /// haze's march reads.
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(crate) fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_layout: &crate::pipelines::GroupLayout,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/post.wgsl"),
            source: wgpu::ShaderSource::Wgsl(POST_SHADER.into()),
        });
        let bloom_format = bloom_format(device);
        // Every pass of post.wgsl shares one layout.
        let layout = pipelines::layout(device, &POST);
        let pipeline_layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::POST_PIPELINES,
            &[&layout],
        );
        let pipeline = |label, entry: crate::shaders::Entry, format: wgpu::TextureFormat, blend| {
            fullscreen(
                device,
                &Fullscreen {
                    label,
                    layout: &pipeline_layout,
                    module: &module,
                    vertex: post::vs_fullscreen,
                    fragment: entry,
                    constants: &[],
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    depth: None,
                },
            )
        };
        let levels_pipeline = |label, entry: crate::shaders::Entry| {
            fullscreen(
                device,
                &Fullscreen {
                    label,
                    layout: &pipeline_layout,
                    module: &module,
                    vertex: post::vs_fullscreen,
                    fragment: entry,
                    constants: &[],
                    targets: &[Some(SCENE_LEVEL_FORMAT.into()), Some(bloom_format.into())],
                    depth: None,
                },
            )
        };
        let add = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        });
        let uniform = crate::buffers::uniform(
            device,
            "engine/post-uniform",
            size_of::<PostUniform>() as u64,
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/post-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            // Blends the blurred scene levels reflections read.
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let haze_layout = pipelines::layout(device, &HAZE);
        let haze_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/haze.wgsl"),
            source: wgpu::ShaderSource::Wgsl(HAZE_SHADER.into()),
        });
        let haze_pipeline_layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::HAZE_PIPELINES,
            &[camera_layout, &haze_layout],
        );
        let march = fullscreen(
            device,
            &Fullscreen {
                label: "engine/post-haze",
                layout: &haze_pipeline_layout,
                module: &haze_module,
                vertex: haze::vs_fullscreen,
                fragment: haze::fs_haze,
                constants: &[],
                targets: &[Some(HAZE_FORMAT.into())],
                depth: None,
            },
        );
        Self {
            layout,
            fillers: Fillers::new(device),
            occlusion: pipeline(
                "engine/post-occlusion",
                post::fs_occlusion,
                RAW_OCCLUSION_FORMAT,
                None,
            ),
            effects_filter: pipeline(
                "engine/post-effects-filter",
                post::fs_effects_filter,
                OCCLUSION_FORMAT,
                None,
            ),
            reflections: pipeline(
                "engine/post-reflections",
                post::fs_reflections,
                REFLECTION_FORMAT,
                None,
            ),
            reflections_filter: pipeline(
                "engine/post-reflections-filter",
                post::fs_reflections_filter,
                REFLECTION_FORMAT,
                None,
            ),
            nearest: DepthPyramid::new(device, Reduction::Nearest),
            trace_index: 0,
            haze: march,
            haze_layout,
            haze_filter: pipeline(
                "engine/post-haze-filter",
                post::fs_haze_filter,
                HAZE_FORMAT,
                None,
            ),
            haze_history: None,
            haze_kept: None,
            haze_frames: 0,
            bloom_first: pipeline(
                "engine/post-bloom-first",
                post::fs_bloom_first,
                bloom_format,
                None,
            ),
            bloom_down: pipeline(
                "engine/post-bloom-down",
                post::fs_bloom_down,
                bloom_format,
                None,
            ),
            bloom_up: pipeline("engine/post-bloom-up", post::fs_bloom_up, bloom_format, add),
            scene_down: pipeline(
                "engine/post-scene-down",
                post::fs_bloom_down,
                SCENE_LEVEL_FORMAT,
                None,
            ),
            bloom_format,
            levels_first: levels_pipeline("engine/post-levels-first", post::fs_levels_first),
            levels_down: levels_pipeline("engine/post-levels-down", post::fs_levels_down),
            composite: pipeline(
                "engine/post-composite",
                post::fs_composite,
                surface_format,
                None,
            ),
            uniform,
            written: None,
            sampler,
            targets: ScreenTarget::default(),
            region: FrameRegion::default(),
            occlusion_group: GroupCache::default(),
            effects_filter_group: GroupCache::default(),
            reflections_group: GroupCache::default(),
            filter_group: GroupCache::default(),
            haze_group: GroupCache::default(),
            haze_filter_group: GroupCache::default(),
        }
    }

    /// Releases the targets while no frame uses an effect, and the haze's
    /// history with them; the next such frame recreates them.
    pub(crate) fn release_targets(&mut self) {
        self.targets.release();
        self.occlusion_group.clear();
        self.effects_filter_group.clear();
        self.reflections_group.clear();
        self.filter_group.clear();
        self.haze_filter_group.clear();
        self.haze_history = None;
        self.nearest.release();
    }

    /// Target bytes: scene (4), glow (2) and surface (4) per pixel, at half
    /// resolution raw and filtered occlusion (8 each), traced and filtered
    /// reflections (4 each) and the haze's march, filtered result and
    /// history (8 each), and the bloom chain and blurred scene levels (8
    /// bytes per texel of their first levels), and the reflections' depth
    /// pyramid.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.targets.size().map_or(0, |size| {
            let pixels = u64::from(size.width) * u64::from(size.height);
            pixels * 10 + pixels / 4 * 48 + pixels / 4 * 8 * 4 / 3 * 2
        }) + self.nearest.capacity_bytes()
    }

    /// Sizes the targets to `region`'s display and writes this frame's
    /// settings: the effects of `snapshot` over the depth buffer `depth`, at
    /// the region's render size. `temporal` frames trace reflections from
    /// another pixel of each 2x2 block than the frame before; `display`
    /// frames filter the haze over the last display frame's
    /// ([`Self::begin_haze`]).
    pub(crate) fn begin(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        snapshot: &RenderSnapshot,
        (region, depth): (FrameRegion, &Versioned<wgpu::TextureView>),
        (temporal, display): (bool, bool),
        sky: [(&'static str, Bound<'_>); 3],
    ) -> Effects {
        let size = region.display;
        if self.targets.resizes(size) {
            let targets = self.create_targets(device, size);
            self.targets.set(size, targets);
            // New targets hold no history of the haze.
            self.haze_history = None;
        }
        self.region = region;
        let frame = Effects::of(&snapshot.environment);
        self.bind_occlusion(device, frame, depth, sky);
        let (history, turn) = self.begin_haze(device, (frame, depth), snapshot, display);
        let targets = self.targets.get().expect("sized above");
        let post = || Bound::fixed(self.uniform.as_entire_binding());
        let target = |view| Bound::of(targets, pipelines::view(view));
        let shared = (&POST, &self.layout);
        self.trace_index = if temporal {
            (self.trace_index + 1) % 4
        } else {
            0
        };
        if frame.reflections {
            self.nearest.prepare(device, queue, depth, region);
        }
        if frame.reflections
            && let Some(nearest) = self.nearest.view()
        {
            self.reflections_group.bind_filled(
                device,
                "engine/post-reflections-group",
                shared,
                [
                    ("post", post()),
                    ("depth_texture", depth.bound()),
                    ("scene", target(&targets.scene)),
                    ("glow", target(&targets.glow)),
                    ("linear", Bound::fixed(pipelines::sampler(&self.sampler))),
                    ("nearest_depth", nearest.bound()),
                    ("surface_texture", target(&targets.surface)),
                    ("scene_levels", target(&targets.scene_levels)),
                ],
                &self.fillers,
            );
            self.filter_group.bind_filled(
                device,
                "engine/post-reflections-filter-group",
                shared,
                [
                    ("post", post()),
                    ("depth_texture", depth.bound()),
                    ("traced", target(&targets.traced)),
                ],
                &self.fillers,
            );
        }
        let (width, height) = (region.render.width as f32, region.render.height as f32);
        let (wide, tall) = (size.width as f32, size.height as f32);
        let view_projection = snapshot.view_projection;
        // Row y of a perspective view-projection is the focal scale times a
        // unit axis of the view: its length converts world units at unit
        // distance into half the viewport height.
        let focal = view_projection.row(1).truncate().length();
        let half = (
            (size.width / 2).max(1) as f32,
            (size.height / 2).max(1) as f32,
        );
        let atmosphere = snapshot.environment.atmosphere.sanitized();
        let toward_sun = atmosphere.sun_direction.normalize_or_zero();
        let effects = frame.uniforms(
            &snapshot.environment,
            &UniformFrame {
                half_texel: [1.0 / half.0, 1.0 / half.1],
                trace_index: self.trace_index,
                haze_turn: turn,
                haze_history: history
                    .map_or([0.0; 2], |last| [last.drawn.0 as f32, last.drawn.1 as f32]),
            },
        );
        let uniform = PostUniform {
            inverse_view_projection: inverse_or_identity(view_projection).to_cols_array_2d(),
            view_projection: view_projection.to_cols_array_2d(),
            eye: snapshot
                .camera_position
                .extend(focal * height * 0.5)
                .to_array(),
            size: [width, height, 1.0 / width, 1.0 / height],
            extent: [wide, tall, 1.0 / wide, 1.0 / tall],
            occlusion: effects.occlusion,
            bloom: effects.bloom,
            reflections: effects.reflections,
            haze: effects.haze,
            previous_view_projection: history
                .map_or(view_projection, |last| last.view_projection)
                .to_cols_array_2d(),
            previous_eye: history
                .map_or(snapshot.camera_position, |last| last.eye)
                .extend(0.0)
                .to_array(),
            sun_direction: toward_sun
                .extend(f32::from(u8::from(snapshot.environment.dither.enabled)))
                .to_array(),
            sky: effects.sky,
            detail: snapshot.detail_eye.extend(0.0).to_array(),
        };
        if self.written != Some(uniform) {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
            self.written = Some(uniform);
        }
        frame
    }

    /// The groups of the occlusion pass (with sky occlusion's maps `sky`)
    /// and of the filter after it, where `frame` runs them.
    fn bind_occlusion(
        &mut self,
        device: &wgpu::Device,
        frame: Effects,
        depth: &Versioned<wgpu::TextureView>,
        sky: [(&'static str, Bound<'_>); 3],
    ) {
        let targets = self.targets.get().expect("sized in begin");
        let post = || Bound::fixed(self.uniform.as_entire_binding());
        let target = |view| Bound::of(targets, pipelines::view(view));
        let shared = (&POST, &self.layout);
        if frame.occlusion || frame.sky {
            let [sky_maps, sky_sampler, sky] = sky;
            self.occlusion_group.bind_filled(
                device,
                "engine/post-occlusion-group",
                shared,
                [
                    ("post", post()),
                    ("depth_texture", depth.bound()),
                    sky_maps,
                    sky_sampler,
                    sky,
                ],
                &self.fillers,
            );
            self.effects_filter_group.bind_filled(
                device,
                "engine/post-effects-filter-group",
                shared,
                [
                    ("post", post()),
                    ("depth_texture", depth.bound()),
                    ("raw_occlusion", target(&targets.raw_occlusion)),
                ],
                &self.fillers,
            );
        }
    }

    /// The groups of the haze's march and of its filter over the depth
    /// buffer `depth`, where `frame` runs it, and what a `display` frame of
    /// `snapshot` leaves the next one's filter, kept once it is submitted:
    /// returns the history this frame's filter reprojects (the last display
    /// frame's, when this one is displayed too) and the turn of its march's
    /// jitter. Frames shown no more than once (exact ones, previews) keep no
    /// history, read none and leave the jitter unturned.
    fn begin_haze(
        &mut self,
        device: &wgpu::Device,
        (frame, depth): (Effects, &Versioned<wgpu::TextureView>),
        snapshot: &RenderSnapshot,
        display: bool,
    ) -> (Option<HazeHistory>, f32) {
        if frame.haze {
            let targets = self.targets.get().expect("sized in begin");
            let post = || Bound::fixed(self.uniform.as_entire_binding());
            let target = |view| Bound::of(targets, pipelines::view(view));
            self.haze_group.bind(
                device,
                "engine/post-haze-group",
                (&HAZE, &self.haze_layout),
                [("post", post()), ("depth_texture", depth.bound())],
            );
            self.haze_filter_group.bind_filled(
                device,
                "engine/post-haze-filter-group",
                (&POST, &self.layout),
                [
                    ("post", post()),
                    ("depth_texture", depth.bound()),
                    ("marched", target(&targets.marched)),
                    ("haze", target(&targets.history)),
                ],
                &self.fillers,
            );
        }
        self.haze_kept = (display && frame.haze).then_some(HazeHistory {
            view_projection: snapshot.view_projection,
            eye: snapshot.camera_position,
            drawn: self.region.half(),
        });
        if self.haze_kept.is_none() {
            return (None, 0.0);
        }
        // The golden ratio's fraction turns each frame's jitter far from the
        // last few's.
        let turn = (self.haze_frames % 1024) as f32 * 0.618_034;
        (self.haze_history, turn - turn.floor())
    }

    // A group of the shared layout binding `resources` by name, filled
    // where they leave bindings out.
    fn group(
        &self,
        device: &wgpu::Device,
        label: &str,
        resources: &[(&str, wgpu::BindingResource<'_>)],
    ) -> wgpu::BindGroup {
        filled_group(device, label, &POST, &self.layout, resources, &self.fillers)
    }

    /// The scene color, glow and surface targets of the opaque pass.
    pub(crate) fn targets(&self) -> (&wgpu::TextureView, &wgpu::TextureView, &wgpu::TextureView) {
        let targets = self.targets.get().expect("sized in begin");
        (&targets.scene, &targets.glow, &targets.surface)
    }

    /// The scene color's texture, which water copies before it draws over
    /// it.
    pub(crate) fn scene_texture(&self) -> &wgpu::Texture {
        &self.targets.get().expect("sized in begin").textures[0]
    }

    /// The scene color as a group binds it, for glass refracting it.
    pub(crate) fn scene(&self) -> Bound<'_> {
        let targets = self.targets.get().expect("sized in begin");
        Bound::of(targets, pipelines::view(&targets.scene))
    }

    /// What the temporal resolve reads, as its group binds them: the scene,
    /// occlusion, bloom, reflections and the haze, and the settings uniform.
    pub(crate) fn resolve_inputs(&self) -> ([Bound<'_>; 5], Bound<'_>) {
        let targets = self.targets.get().expect("sized in begin");
        let target = |view| Bound::of(targets, pipelines::view(view));
        (
            [
                target(&targets.scene),
                target(&targets.occlusion),
                target(&targets.bloom[0]),
                target(&targets.reflections),
                target(&targets.haze),
            ],
            Bound::fixed(self.uniform.as_entire_binding()),
        )
    }

    /// Records ambient occlusion, bloom, reflections, the haze (its march
    /// over the camera group `camera` of the main passes, then its filter)
    /// and the filter of occlusion after the opaque and transparent passes.
    /// Frames with bloom and reflections blur the scene levels reflections
    /// read in the bloom's passes down. `profiler` times occlusion's samples
    /// (`ambient_occlusion`), bloom with those scene levels (`bloom`),
    /// reflections from their depth pyramid to their filter (`reflections`),
    /// the haze's march and filter (`haze`) and occlusion's filter
    /// (`effects_filter`).
    fn encode_effects(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame: Effects,
        camera: &wgpu::BindGroup,
        profiler: &GpuProfiler,
    ) {
        let targets = self.targets.get().expect("sized in begin");
        // Half the render size of the half-resolution targets.
        let half = self.region.half();
        if (frame.occlusion || frame.sky)
            && let Some(group) = self.occlusion_group.get()
        {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-occlusion",
                (&targets.raw_occlusion, half),
                wgpu::LoadOp::Load,
                profiler.span("ambient_occlusion").map(Span::render),
            );
            pass.set_pipeline(&self.occlusion);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
        let reflecting = frame.reflections
            && self.reflections_group.get().is_some()
            && self.filter_group.get().is_some();
        // Bloom first: with reflections, its passes down also blur the scene
        // for rough reflections.
        let levels_done = if frame.bloom {
            self.encode_bloom(encoder, reflecting, profiler)
        } else {
            0
        };
        if reflecting
            && let Some(group) = self.reflections_group.get()
            && let Some(filter) = self.filter_group.get()
        {
            self.nearest.encode(encoder, profiler.span("reflections"));
            // The scene blurred down its levels for rough reflections, past
            // those the bloom's passes wrote.
            for (level, (view, group)) in targets
                .scene_level_views
                .iter()
                .zip(&targets.scene_level_groups)
                .enumerate()
                .skip(levels_done)
            {
                let mut pass = fullscreen_pass(
                    encoder,
                    "engine/post-scene-levels",
                    (view, self.level_region(level as u32, targets.levels)),
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    None,
                );
                pass.set_pipeline(&self.scene_down);
                pass.set_bind_group(0, group, &[]);
                pass.draw(0..3, 0..1);
            }
            {
                let mut pass = fullscreen_pass(
                    encoder,
                    "engine/post-reflections",
                    (&targets.traced, half),
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    None,
                );
                pass.set_pipeline(&self.reflections);
                pass.set_bind_group(0, group, &[]);
                pass.draw(0..3, 0..1);
            }
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-reflections-filter",
                (&targets.reflections, half),
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                profiler
                    .span("reflections")
                    .map(|span| span.render_part(false, true)),
            );
            pass.set_pipeline(&self.reflections_filter);
            pass.set_bind_group(0, filter, &[]);
            pass.draw(0..3, 0..1);
        }
        if frame.haze {
            self.encode_haze(encoder, camera, profiler);
        }
        // The occlusion filtered at half resolution.
        if (frame.occlusion || frame.sky)
            && let Some(group) = self.effects_filter_group.get()
        {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-effects-filter",
                (&targets.occlusion, half),
                wgpu::LoadOp::Load,
                profiler.span("effects_filter").map(Span::render),
            );
            pass.set_pipeline(&self.effects_filter);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Records the haze: its march at half resolution over the camera group
    /// `camera` of the main passes, its filter over time, then the filtered
    /// result copied into the history the next frame's filter reprojects.
    /// `profiler` times the march and the filter (`haze`).
    fn encode_haze(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera: &wgpu::BindGroup,
        profiler: &GpuProfiler,
    ) {
        let (Some(march), Some(filter)) = (self.haze_group.get(), self.haze_filter_group.get())
        else {
            return;
        };
        let targets = self.targets.get().expect("sized in begin");
        let half = self.region.half();
        {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-haze",
                (&targets.marched, half),
                wgpu::LoadOp::Load,
                profiler
                    .span("haze")
                    .map(|span| span.render_part(true, false)),
            );
            pass.set_pipeline(&self.haze);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(1, march, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-haze-filter",
                (&targets.haze, half),
                wgpu::LoadOp::Load,
                profiler
                    .span("haze")
                    .map(|span| span.render_part(false, true)),
            );
            pass.set_pipeline(&self.haze_filter);
            pass.set_bind_group(0, filter, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_texture(
            targets.textures[HAZE_TEXTURE].as_image_copy(),
            targets.textures[HAZE_HISTORY].as_image_copy(),
            wgpu::Extent3d {
                width: half.0,
                height: half.1,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Records the bloom chain down and back up. With `scene_levels` its
    /// passes down also halve the scene levels rough reflections read, as
    /// many as both chains have; returns how many they wrote.
    fn encode_bloom(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_levels: bool,
        profiler: &GpuProfiler,
    ) -> usize {
        let targets = self.targets.get().expect("sized in begin");
        let levels = targets.bloom.len();
        // The first pass down begins the span, the last pass up ends it.
        let span = profiler.span("bloom");
        let timed = |first: bool, last: bool| {
            span.filter(|_| first || last)
                .map(|span| span.render_part(first, last))
        };
        let shared = if scene_levels {
            targets.scene_level_views.len().min(levels)
        } else {
            0
        };
        let drawn = |level: usize| self.level_region(level as u32, targets.levels);
        if shared > 0 {
            for level in 0..shared {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("engine/post-levels"),
                    color_attachments: &[
                        Some(overwritten(&targets.scene_level_views[level])),
                        Some(overwritten(&targets.bloom[level])),
                    ],
                    depth_stencil_attachment: None,
                    timestamp_writes: timed(level == 0, levels == 1),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                draw_within(&mut pass, drawn(level));
                if level == 0 {
                    pass.set_pipeline(&self.levels_first);
                    pass.set_bind_group(0, &targets.levels_first, &[]);
                } else {
                    pass.set_pipeline(&self.levels_down);
                    pass.set_bind_group(0, &targets.levels_down[level - 1], &[]);
                }
                pass.draw(0..3, 0..1);
            }
        } else {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-bloom",
                (&targets.bloom[0], drawn(0)),
                wgpu::LoadOp::Load,
                timed(true, levels == 1),
            );
            pass.set_pipeline(&self.bloom_first);
            pass.set_bind_group(0, &targets.bloom_first, &[]);
            pass.draw(0..3, 0..1);
        }
        for level in shared.max(1)..levels {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-bloom",
                (&targets.bloom[level], drawn(level)),
                wgpu::LoadOp::Load,
                None,
            );
            pass.set_pipeline(&self.bloom_down);
            pass.set_bind_group(0, &targets.bloom_down[level - 1], &[]);
            pass.draw(0..3, 0..1);
        }
        for level in (0..levels - 1).rev() {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-bloom",
                (&targets.bloom[level], drawn(level)),
                wgpu::LoadOp::Load,
                timed(false, level == 0),
            );
            pass.set_pipeline(&self.bloom_up);
            pass.set_bind_group(0, &targets.bloom_up[level], &[]);
            pass.draw(0..3, 0..1);
        }
        shared
    }

    /// The texels the frame draws of level `level` of a half-resolution
    /// chain whose first level is `first` texels: half the render size,
    /// halved `level` times, a texel at least, and one more, which shaders
    /// reading within the share of the level the render size is of the
    /// display take, however their arithmetic rounds; within the level.
    fn level_region(&self, level: u32, first: (u32, u32)) -> (u32, u32) {
        let (wide, tall) = self.region.half();
        let part = |drawn: u32, whole: u32| {
            let whole = (whole >> level).max(1);
            ((drawn >> level).max(1) + 1).min(whole)
        };
        (part(wide, first.0), part(tall, first.1))
    }

    /// Draws the composite of a frame without temporal antialiasing into
    /// `pass`, its node's pass over the canvas, upscaling the scene.
    fn draw_composite(&self, mut pass: wgpu::RenderPass<'_>) {
        let targets = self.targets.get().expect("sized in begin");
        pass.set_pipeline(&self.composite);
        pass.set_bind_group(0, &targets.composite, &[]);
        pass.draw(0..3, 0..1);
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn create_targets(
        &self,
        device: &wgpu::Device,
        FrameSize { width, height }: FrameSize,
    ) -> Targets {
        let half = ((width / 2).max(1), (height / 2).max(1));
        let texture = |label, (width, height): (u32, u32), format, levels| {
            crate::targets::texture_2d(
                device,
                label,
                (width, height),
                1,
                levels,
                format,
                crate::targets::RENDER_TARGET,
            )
        };
        let haze_texture = |label, usage| {
            crate::targets::texture_2d(device, label, half, 1, 1, HAZE_FORMAT, usage)
        };
        let levels = BLOOM_LEVELS.min(crate::gpu_data::mip_levels(half.0, half.1));
        let scene_levels = SCENE_LEVELS.min(crate::gpu_data::mip_levels(half.0, half.1));
        let textures = [
            // Water copies the scene it refracts before it draws over it.
            crate::targets::texture_2d(
                device,
                "engine/post-scene",
                (width, height),
                1,
                1,
                SCENE_FORMAT,
                crate::targets::RENDER_TARGET | wgpu::TextureUsages::COPY_SRC,
            ),
            texture("engine/post-glow", (width, height), GLOW_FORMAT, 1),
            texture("engine/post-occlusion", half, OCCLUSION_FORMAT, 1),
            texture("engine/post-bloom", half, self.bloom_format, levels),
            texture("engine/post-reflections", half, REFLECTION_FORMAT, 1),
            texture("engine/post-haze-march", half, HAZE_FORMAT, 1),
            texture("engine/post-reflections-traced", half, REFLECTION_FORMAT, 1),
            texture("engine/post-occlusion-raw", half, RAW_OCCLUSION_FORMAT, 1),
            texture("engine/post-surface", (width, height), SURFACE_FORMAT, 1),
            texture(
                "engine/post-scene-levels",
                half,
                SCENE_LEVEL_FORMAT,
                scene_levels,
            ),
            // The filtered haze is copied into the history the next frame
            // reads.
            haze_texture(
                "engine/post-haze",
                crate::targets::RENDER_TARGET | wgpu::TextureUsages::COPY_SRC,
            ),
            haze_texture(
                "engine/post-haze-history",
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            ),
        ];
        let [
            scene,
            glow,
            occlusion,
            _,
            reflections,
            marched,
            traced,
            raw_occlusion,
            surface,
            scene_levels_view,
            haze,
            history,
        ] = textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
        let bloom: Vec<wgpu::TextureView> = (0..levels)
            .map(|level| {
                textures[3].create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let uniform = ("post", self.uniform.as_entire_binding());
        let sampler = ("linear", pipelines::sampler(&self.sampler));
        let group = |resources: &[(&str, wgpu::BindingResource<'_>)]| {
            self.group(device, "engine/post-group", resources)
        };
        let bloom_first = group(&[
            uniform.clone(),
            texture_entry("scene", &scene),
            texture_entry("glow", &glow),
            sampler.clone(),
        ]);
        let bloom_down = (0..bloom.len().saturating_sub(1))
            .map(|level| group(&[texture_entry("source", &bloom[level]), sampler.clone()]))
            .collect();
        let bloom_up = (0..bloom.len().saturating_sub(1))
            .map(|level| group(&[texture_entry("source", &bloom[level + 1]), sampler.clone()]))
            .collect();
        // The scene levels blur with the bloom's downsampling filter, the
        // first from the scene itself.
        let scene_level_views: Vec<wgpu::TextureView> = (0..scene_levels)
            .map(|level| {
                textures[9].create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let scene_level_groups = (0..scene_level_views.len())
            .map(|level| {
                let source = if level == 0 {
                    &scene
                } else {
                    &scene_level_views[level - 1]
                };
                group(&[texture_entry("source", source), sampler.clone()])
            })
            .collect();
        let levels_first = group(&[
            uniform.clone(),
            texture_entry("scene", &scene),
            texture_entry("glow", &glow),
            sampler.clone(),
        ]);
        let levels_down = (1..scene_level_views.len().min(bloom.len()))
            .map(|level| {
                group(&[
                    texture_entry("source", &scene_level_views[level - 1]),
                    texture_entry("bloom_source", &bloom[level - 1]),
                    sampler.clone(),
                ])
            })
            .collect();
        let composite = group(&[
            uniform.clone(),
            texture_entry("scene", &scene),
            texture_entry("occlusion", &occlusion),
            texture_entry("bloom", &bloom[0]),
            sampler.clone(),
            texture_entry("reflections", &reflections),
            texture_entry("haze", &haze),
        ]);
        Targets {
            textures,
            scene,
            glow,
            surface,
            levels: half,
            scene_levels: scene_levels_view,
            scene_level_views,
            scene_level_groups,
            occlusion,
            raw_occlusion,
            reflections,
            traced,
            marched,
            haze,
            history,
            bloom,
            bloom_first,
            bloom_down,
            bloom_up,
            levels_first,
            levels_down,
            composite,
        }
    }
}

/// An attachment of `view` that a full-screen pass writes whole: loaded,
/// as nothing needs clearing, and stored.
fn overwritten(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        },
    }
}

// A texture view bound as binding `name`.
fn texture_entry<'a>(
    name: &'a str,
    view: &'a wgpu::TextureView,
) -> (&'a str, wgpu::BindingResource<'a>) {
    (name, pipelines::view(view))
}

pub(crate) fn inverse_or_identity(matrix: Mat4) -> Mat4 {
    let inverse = matrix.inverse();
    if inverse.is_finite() {
        inverse
    } else {
        Mat4::IDENTITY
    }
}

/// A pass labeled `label` drawing `view`'s first `drawn` texels, from its
/// top-left corner.
fn fullscreen_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    label: &str,
    (view, drawn): (&wgpu::TextureView, (u32, u32)),
    load: wgpu::LoadOp<wgpu::Color>,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) -> wgpu::RenderPass<'e> {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    draw_within(&mut pass, drawn);
    pass
}

/// Which screen effects a frame applied.
pub(crate) struct EffectStats {
    pub(crate) ambient_occlusion: Applied,
    pub(crate) bloom: Applied,
    pub(crate) haze: Applied,
    pub(crate) reflections: Applied,
}

impl RenderFeature for PostPass {
    type Frame = Effects;
    type Stats = EffectStats;

    const LABELS: &'static [&'static str] = &["engine/post-", "engine/post.", "engine/haze."];

    /// The effects that run read what they need into their results, the
    /// haze into its history too, which the next frame reads; the composite
    /// (frames drawing into the effects' targets without temporal
    /// antialiasing) lays them with the scene over the canvas.
    fn declare(effects: &Effects, shape: &FrameShape, graph: &mut RenderGraph) {
        let targets = shape.targets;
        if targets != crate::render_graph::Targets::Canvas && effects.any() {
            effects
                .reads(graph.add(Pass::Feature(FeaturePass::Effects)))
                .write(Resource::Effects)
                .write_if(effects.haze, Resource::Haze);
        }
        if targets == crate::render_graph::Targets::Post {
            graph
                .add(Pass::Feature(FeaturePass::Composite))
                .read(Resource::Scene)
                .read_if(effects.any(), Resource::Effects)
                .color(Resource::Canvas, Access::Clear(Clear::Color(BLACK)));
        }
    }

    fn encode(
        &self,
        effects: &Effects,
        node: &Node,
        encode: &Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        if node.pass() == Pass::Feature(FeaturePass::Composite) {
            let pass = encode.render_pass(
                encoder,
                node,
                "engine/post-composite",
                encode.profiler.span("post_composite").map(Span::render),
            );
            self.draw_composite(pass);
        } else {
            // Ambient occlusion, bloom, reflections and the haze, after the
            // opaque and transparent passes.
            self.encode_effects(encoder, *effects, encode.camera, encode.profiler);
        }
        0
    }

    /// A submitted display frame's haze is the history the next frame's
    /// filter reprojects; any other frame leaves none.
    fn after_submit(&mut self, frame: Option<&Effects>) {
        let kept = self.haze_kept.take();
        self.haze_history = frame.and(kept);
        if self.haze_history.is_some() {
            self.haze_frames = self.haze_frames.wrapping_add(1);
        }
    }

    fn capacity_bytes(&self) -> u64 {
        PostPass::capacity_bytes(self)
    }

    fn stats(
        _this: Option<&Self>,
        effects: Option<&Effects>,
        _snapshot: &RenderSnapshot,
    ) -> EffectStats {
        let applied = |effect: fn(&Effects) -> bool| Applied {
            active: effects.is_some_and(effect),
        };
        EffectStats {
            ambient_occlusion: applied(|effects| effects.occlusion),
            bloom: applied(|effects| effects.bloom),
            haze: applied(|effects| effects.haze),
            reflections: applied(|effects| effects.reflections),
        }
    }
}
