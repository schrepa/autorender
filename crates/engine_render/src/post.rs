//! Screen effects: ambient occlusion and bloom, sun shafts
//! and screen-space reflections.
//!
//! Frames with an effect draw the scene into a color target whose alpha is the
//! ambient share of each pixel's light, and a glow target of each pixel's
//! exposed brightness above the bloom threshold and its reflectivity. Ambient
//! occlusion samples the depth buffer at half resolution; sun shafts march
//! the far shadow maps at quarter resolution; one half-resolution filter then
//! smooths the occlusion on each surface and brings the shafts up beside it.
//! Reflections walk a pyramid of the depth buffer's nearest depths at half
//! resolution and filter the result; bloom filters the glowing light down a
//! chain of half-resolution levels and back up. Temporally antialiased frames
//! apply them in their resolve, other frames in a composite
//! that writes the canvas.

use crate::shaders::entry::{post, shafts};
use engine_core::{Applied, RenderSnapshot, glam::Mat4};

use crate::bindings::{POST, SHAFTS};
use crate::cached::{Bound, GroupCache, Versioned};
use crate::depth_pyramid::{DepthPyramid, Reduction};
use crate::gpu_data::{POST_SHADER, PostUniform, SHAFTS_SHADER};
use crate::pipelines::{self, Fillers, Fullscreen, filled_group, fullscreen};
use crate::profiling::{GpuProfiler, Span};
use crate::render_features::{Encode, FeaturePass, RenderFeature, color_operations};
use crate::render_graph::{Access, BLACK, Clear, FrameShape, Node, Pass, RenderGraph, Resource};
use crate::screen_effects::{Effects, UniformFrame};
use crate::targets::{FrameSize, ScreenTarget};

/// Linear color with 8 perceptual bits; alpha holds the ambient share.
pub(crate) const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Glow weight and reflectivity.
pub(crate) const GLOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;
/// The surface reflections follow: the shading normal,
/// octahedral, and roughness, 10 bits each.
pub(crate) const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgb10a2Unorm;
/// The filtered occlusion's visibility and the sun shafts' light, which the
/// composite and resolve read with the same taps.
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
/// Scattered sunlight and the length of the ray that
/// gathered it, at quarter resolution.
const SHAFT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
/// Half-resolution bloom levels; each halves the one before.
const BLOOM_LEVELS: u32 = 6;
/// Blurred half-resolution levels of the scene rough reflections read.
const SCENE_LEVELS: u32 = 6;

struct Targets {
    // Scene, glow, occlusion (with shafts), bloom, reflections, traced
    // shafts, traced reflections, raw occlusion, surface and scene levels,
    // destroyed with the targets.
    textures: [wgpu::Texture; 10],
    scene: wgpu::TextureView,
    glow: wgpu::TextureView,
    surface: wgpu::TextureView,
    // Every blurred scene level, which reflections read, and group `i`
    // reading the scene (0) or level `i - 1` to write level `i`.
    scene_levels: wgpu::TextureView,
    scene_level_views: Vec<wgpu::TextureView>,
    scene_level_groups: Vec<wgpu::BindGroup>,
    // Filtered occlusion and shafts, which the composite and resolve read,
    // and the raw occlusion and traced shafts the filter reads.
    occlusion: wgpu::TextureView,
    raw_occlusion: wgpu::TextureView,
    // Filtered reflections, which the composite and resolve read, and the
    // traced ones the filter reads.
    reflections: wgpu::TextureView,
    traced: wgpu::TextureView,
    shafts: wgpu::TextureView,
    // One view per bloom level, and the level count.
    bloom: Vec<wgpu::TextureView>,
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
    // Filters occlusion and brings shafts to half resolution in one pass.
    effects_filter: wgpu::RenderPipeline,
    reflections: wgpu::RenderPipeline,
    reflections_filter: wgpu::RenderPipeline,
    // Nearest depths the reflection rays walk, created with the first frame
    // that reflects.
    nearest: DepthPyramid,
    // The pixel of each 2x2 block this frame's reflections trace.
    trace_index: u32,
    // Sun shafts read the camera group of the main passes (shadow maps and
    // lighting) and their own.
    shafts: wgpu::RenderPipeline,
    shafts_layout: crate::pipelines::GroupLayout,
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
    // Occlusion, filter, reflection and shaft bindings over the depth
    // buffer and targets (and reflections', the pyramid).
    occlusion_group: GroupCache<2>,
    effects_filter_group: GroupCache<4>,
    reflections_group: GroupCache<8>,
    filter_group: GroupCache<3>,
    shafts_group: GroupCache<2>,
}

impl PostPass {
    /// `camera_layout` is the first group of the main passes, which sun
    /// shafts read.
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
        let shafts_layout = pipelines::layout(device, &SHAFTS);
        let shafts_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/shafts.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHAFTS_SHADER.into()),
        });
        let shafts_pipeline_layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::SHAFT_PIPELINES,
            &[camera_layout, &shafts_layout],
        );
        let shafts = fullscreen(
            device,
            &Fullscreen {
                label: "engine/post-shafts",
                layout: &shafts_pipeline_layout,
                module: &shafts_module,
                vertex: shafts::vs_fullscreen,
                fragment: shafts::fs_shafts,
                constants: &[],
                targets: &[Some(SHAFT_FORMAT.into())],
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
            shafts,
            shafts_layout,
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
            occlusion_group: GroupCache::default(),
            effects_filter_group: GroupCache::default(),
            reflections_group: GroupCache::default(),
            filter_group: GroupCache::default(),
            shafts_group: GroupCache::default(),
        }
    }

    /// Releases the targets while no frame uses an effect; the next such
    /// frame recreates them.
    pub(crate) fn release_targets(&mut self) {
        self.targets.release();
        self.occlusion_group.clear();
        self.effects_filter_group.clear();
        self.reflections_group.clear();
        self.filter_group.clear();
        self.nearest.release();
    }

    /// Target bytes: scene (4), glow (2) and surface (4) per pixel, at half
    /// resolution raw and filtered occlusion (4 each, the filtered with the
    /// shafts), traced and filtered reflections (8 each), at quarter
    /// resolution the traced shafts (4), and the bloom chain and blurred
    /// scene levels (8 bytes per texel of their first levels), and the
    /// reflections' depth pyramid.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.targets.size().map_or(0, |size| {
            let pixels = u64::from(size.width) * u64::from(size.height);
            pixels * 10 + pixels / 4 * 24 + pixels / 16 * 4 + pixels / 4 * 8 * 4 / 3 * 2
        }) + self.nearest.capacity_bytes()
    }

    /// Sizes the targets to `size` and writes this frame's settings: the
    /// effects of `snapshot` over the depth buffer `depth`.
    /// `temporal` frames trace reflections from another pixel of each 2x2
    /// block than the frame before.
    pub(crate) fn begin(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        snapshot: &RenderSnapshot,
        size: FrameSize,
        depth: &Versioned<wgpu::TextureView>,
        temporal: bool,
    ) -> Effects {
        if self.targets.resizes(size) {
            let targets = self.create_targets(device, size);
            self.targets.set(size, targets);
        }
        let frame = Effects::of(&snapshot.environment);
        let targets = self.targets.get().expect("sized above");
        let post = || Bound::fixed(self.uniform.as_entire_binding());
        let target = |view| Bound::of(targets, pipelines::view(view));
        let shared = (&POST, &self.layout);
        if frame.occlusion || frame.contact {
            self.occlusion_group.bind_filled(
                device,
                "engine/post-occlusion-group",
                shared,
                [("post", post()), ("depth_texture", depth.bound())],
                &self.fillers,
            );
        }
        if frame.occlusion || frame.shafts || frame.contact {
            self.effects_filter_group.bind_filled(
                device,
                "engine/post-effects-filter-group",
                shared,
                [
                    ("post", post()),
                    ("depth_texture", depth.bound()),
                    ("shafts", target(&targets.shafts)),
                    ("raw_occlusion", target(&targets.raw_occlusion)),
                ],
                &self.fillers,
            );
        }
        self.trace_index = if temporal {
            (self.trace_index + 1) % 4
        } else {
            0
        };
        if frame.reflections {
            self.nearest.prepare(device, depth, size);
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
        if frame.shafts {
            self.shafts_group.bind(
                device,
                "engine/post-shafts-group",
                (&SHAFTS, &self.shafts_layout),
                [("post", post()), ("depth_texture", depth.bound())],
            );
        }
        let (width, height) = (size.width as f32, size.height as f32);
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
            occlusion: effects.occlusion,
            bloom: effects.bloom,
            reflections: effects.reflections,
            shafts: effects.shafts,
            sun: (atmosphere.sun_color * atmosphere.exposure)
                .extend(0.0)
                .to_array(),
            contact: effects.contact,
            sun_direction: toward_sun.extend(0.0).to_array(),
        };
        if self.written != Some(uniform) {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
            self.written = Some(uniform);
        }
        frame
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

    /// The scene color as a group binds it, for glass refracting it.
    pub(crate) fn scene(&self) -> Bound<'_> {
        let targets = self.targets.get().expect("sized in begin");
        Bound::of(targets, pipelines::view(&targets.scene))
    }

    /// What the temporal resolve reads, as its group binds them: the scene,
    /// occlusion with shafts, bloom and reflections, and the settings
    /// uniform.
    pub(crate) fn resolve_inputs(&self) -> ([Bound<'_>; 4], Bound<'_>) {
        let targets = self.targets.get().expect("sized in begin");
        let target = |view| Bound::of(targets, pipelines::view(view));
        (
            [
                target(&targets.scene),
                target(&targets.occlusion),
                target(&targets.bloom[0]),
                target(&targets.reflections),
            ],
            Bound::fixed(self.uniform.as_entire_binding()),
        )
    }

    /// Records ambient occlusion, bloom, reflections, sun shafts (over the
    /// camera group `camera` of the main passes) and the filter of occlusion
    /// and shafts after the opaque and transparent passes. Frames with bloom
    /// and reflections blur the scene levels reflections read in the bloom's
    /// passes down. `profiler` times occlusion's samples
    /// (`ambient_occlusion`), bloom with those scene levels (`bloom`),
    /// reflections from their depth pyramid to their filter (`reflections`),
    /// shafts (`sun_shafts`) and the filter (`effects_filter`).
    fn encode_effects(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame: Effects,
        camera: &wgpu::BindGroup,
        profiler: &GpuProfiler,
    ) {
        let targets = self.targets.get().expect("sized in begin");
        if (frame.occlusion || frame.contact)
            && let Some(group) = self.occlusion_group.get()
        {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-occlusion",
                &targets.raw_occlusion,
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
            for (view, group) in targets
                .scene_level_views
                .iter()
                .zip(&targets.scene_level_groups)
                .skip(levels_done)
            {
                let mut pass = fullscreen_pass(
                    encoder,
                    "engine/post-scene-levels",
                    view,
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
                    &targets.traced,
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
                &targets.reflections,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                profiler
                    .span("reflections")
                    .map(|span| span.render_part(false, true)),
            );
            pass.set_pipeline(&self.reflections_filter);
            pass.set_bind_group(0, filter, &[]);
            pass.draw(0..3, 0..1);
        }
        if frame.shafts
            && let Some(group) = self.shafts_group.get()
        {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-shafts",
                &targets.shafts,
                wgpu::LoadOp::Load,
                profiler.span("sun_shafts").map(Span::render),
            );
            pass.set_pipeline(&self.shafts);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(1, group, &[]);
            pass.draw(0..3, 0..1);
        }
        // The occlusion and contact shadows filtered and the shafts at half
        // resolution, together (1, 0 and 1 where off).
        if (frame.occlusion || frame.shafts || frame.contact)
            && let Some(group) = self.effects_filter_group.get()
        {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-effects-filter",
                &targets.occlusion,
                wgpu::LoadOp::Load,
                profiler.span("effects_filter").map(Span::render),
            );
            pass.set_pipeline(&self.effects_filter);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
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
                &targets.bloom[0],
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
                &targets.bloom[level],
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
                &targets.bloom[level],
                wgpu::LoadOp::Load,
                timed(false, level == 0),
            );
            pass.set_pipeline(&self.bloom_up);
            pass.set_bind_group(0, &targets.bloom_up[level], &[]);
            pass.draw(0..3, 0..1);
        }
        shared
    }

    /// Records the composite of a frame without temporal antialiasing into
    /// `canvas` with operations `ops`, timed by `profiler`
    /// (`post_composite`).
    fn encode_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        canvas: &wgpu::TextureView,
        ops: wgpu::Operations<wgpu::Color>,
        profiler: &GpuProfiler,
    ) {
        let targets = self.targets.get().expect("sized in begin");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/post-composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: canvas,
                depth_slice: None,
                resolve_target: None,
                ops,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: profiler.span("post_composite").map(Span::render),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.composite);
        pass.set_bind_group(0, &targets.composite, &[]);
        pass.draw(0..3, 0..1);
    }

    fn create_targets(
        &self,
        device: &wgpu::Device,
        FrameSize { width, height }: FrameSize,
    ) -> Targets {
        let half = ((width / 2).max(1), (height / 2).max(1));
        let quarter = (width.div_ceil(4).max(1), height.div_ceil(4).max(1));
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
        let levels = BLOOM_LEVELS.min(crate::gpu_data::mip_levels(half.0, half.1));
        let scene_levels = SCENE_LEVELS.min(crate::gpu_data::mip_levels(half.0, half.1));
        let textures = [
            texture("engine/post-scene", (width, height), SCENE_FORMAT, 1),
            texture("engine/post-glow", (width, height), GLOW_FORMAT, 1),
            texture("engine/post-occlusion", half, OCCLUSION_FORMAT, 1),
            texture("engine/post-bloom", half, self.bloom_format, levels),
            texture("engine/post-reflections", half, REFLECTION_FORMAT, 1),
            texture("engine/post-shafts", quarter, SHAFT_FORMAT, 1),
            texture("engine/post-reflections-traced", half, REFLECTION_FORMAT, 1),
            texture("engine/post-occlusion-raw", half, RAW_OCCLUSION_FORMAT, 1),
            texture("engine/post-surface", (width, height), SURFACE_FORMAT, 1),
            texture(
                "engine/post-scene-levels",
                half,
                SCENE_LEVEL_FORMAT,
                scene_levels,
            ),
        ];
        let [
            scene,
            glow,
            occlusion,
            _,
            reflections,
            shafts,
            traced,
            raw_occlusion,
            surface,
            scene_levels_view,
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
        ]);
        Targets {
            textures,
            scene,
            glow,
            surface,
            scene_levels: scene_levels_view,
            scene_level_views,
            scene_level_groups,
            occlusion,
            raw_occlusion,
            reflections,
            traced,
            shafts,
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

fn inverse_or_identity(matrix: Mat4) -> Mat4 {
    let inverse = matrix.inverse();
    if inverse.is_finite() {
        inverse
    } else {
        Mat4::IDENTITY
    }
}

fn fullscreen_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    label: &str,
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
    })
}

/// Which screen effects a frame applied.
pub(crate) struct EffectStats {
    pub(crate) ambient_occlusion: Applied,
    pub(crate) contact_shadows: Applied,
    pub(crate) bloom: Applied,
    pub(crate) sun_shafts: Applied,
    pub(crate) reflections: Applied,
}

impl RenderFeature for PostPass {
    type Frame = Effects;
    type Stats = EffectStats;

    /// The effects that run read what they need into their results, which
    /// the composite (frames drawing into the effects' targets without
    /// temporal antialiasing) lays with the scene over the canvas.
    fn declare(effects: &Effects, shape: &FrameShape, graph: &mut RenderGraph) {
        let targets = shape.targets;
        if targets != crate::render_graph::Targets::Canvas && effects.any() {
            effects
                .reads(graph.add(Pass::Feature(FeaturePass::Effects)))
                .write(Resource::Effects);
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
            let (_, canvas) = node
                .colors()
                .next()
                .expect("the composite draws the canvas");
            self.encode_composite(
                encoder,
                encode.view(Resource::Canvas),
                color_operations(canvas),
                encode.profiler,
            );
        } else {
            // Ambient occlusion, bloom, reflections and sun shafts, after the
            // opaque and transparent passes.
            self.encode_effects(encoder, *effects, encode.camera, encode.profiler);
        }
        0
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
            contact_shadows: applied(|effects| effects.contact),
            bloom: applied(|effects| effects.bloom),
            sun_shafts: applied(|effects| effects.shafts),
            reflections: applied(|effects| effects.reflections),
        }
    }
}
