#![expect(
    clippy::expect_used,
    reason = "encoding reads the targets the frame's preparation sized"
)]

use crate::shaders::entry::{haze, post, reflections};
use engine_core::{Applied, RenderSnapshot, RenderStats, glam::Mat4};

use crate::bindings::{GroupRef, HAZE, POST, REFLECTIONS};
use crate::cached::{Bound, GroupCache, Versioned};
use crate::composite::{
    PLAIN_RESOLVE, PixelEffect, PostStage, RESOLVES, composite_entry, composite_path,
    composite_source, pixel_effects,
};
use crate::depth_pyramid::{DepthPyramid, Reduction};
use crate::gpu_data::PostUniform;
use crate::graph::{
    Access, BLACK, Clear, FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl,
    ResourceId, Stage,
};
use crate::lazy::Lazy;
use crate::parts::{
    Begin, BeginOut, Create, Encode, GpuPart, PartDecl, PartEntry, Parts, Prelude, Step, View,
};
use crate::pipeline_layouts::{HAZE_PIPELINES, POST_PIPELINES, REFLECTION_PIPELINES};
use crate::pipelines::{self, Fillers, Fullscreen, filled_group, fullscreen};
use crate::profiling::{GpuProfiler, Span};
use crate::screen_effects::{Effects, UniformFrame};
use crate::targets::{FrameRegion, FrameSize, ScreenTarget, draw_within};

pub(crate) const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
pub(crate) const GLOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg8Unorm;
pub(crate) const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Uint;
const OCCLUSION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
const RAW_OCCLUSION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
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
const REFLECTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const HAZE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const BLOOM_LEVELS: u32 = 6;
const SCENE_LEVELS: u32 = 6;

const HAZE_TEXTURE: usize = 8;
const HAZE_HISTORY: usize = 9;

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "post",
        brick: "post_effects",
        passes: &[
            PassDecl {
                name: "effects",
                stage: Stage::PostScene,
            },
            PassDecl {
                name: "composite",
                stage: Stage::Composite,
            },
        ],
        resources: &[
            ResourceDecl {
                name: "scene",
                lifetime: Lifetime::Frame,
            },
            ResourceDecl {
                name: "glow",
                lifetime: Lifetime::Frame,
            },
            ResourceDecl {
                name: "surface",
                lifetime: Lifetime::Frame,
            },
            ResourceDecl {
                name: "effects",
                lifetime: Lifetime::Frame,
            },
            ResourceDecl {
                name: "haze",
                lifetime: Lifetime::Kept,
            },
        ],
        hooks: &[],
        contributes: &[],
        effects: &[
            PixelEffect {
                stage: PostStage::Light,
                module: "post::haze",
            },
            PixelEffect {
                stage: PostStage::Light,
                module: "post::bloom",
            },
            PixelEffect {
                stage: PostStage::Display,
                module: "post::dither",
            },
        ],
        labels: &["engine/post-", "engine/post.", "engine/haze."],
        lazy: true,
        faces: &[],
        needs: &[],
        piece: None,
    },
    create,
};

const EFFECTS_PASS: usize = 0;
const COMPOSITE_PASS: usize = 1;

fn create(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(PostPass::default())
}

#[derive(Clone, Copy, Default)]
struct Ids {
    effects_pass: Option<PassId>,
    composite_pass: Option<PassId>,
    canvas: Option<ResourceId>,
    scene: Option<ResourceId>,
    effects: Option<ResourceId>,
    haze: Option<ResourceId>,
}

#[derive(Default)]
pub(crate) struct PostPass {
    post: Lazy<Post>,
    frame: Option<Effects>,
    ids: Ids,
    linked: bool,
}

impl PostPass {
    pub(crate) fn active(&self) -> bool {
        self.frame.is_some()
    }

    pub(crate) fn scene(&self) -> Option<Bound<'_>> {
        self.post.get()?.scene()
    }

    pub(crate) fn scene_texture(&self) -> Option<&wgpu::Texture> {
        self.post.get()?.scene_texture()
    }

    pub(crate) fn resolve_inputs(&self) -> Option<([Bound<'_>; 5], Bound<'_>)> {
        self.post.get()?.resolve_inputs()
    }
}

fn link_composites(device: &wgpu::Device, parts: &Parts) {
    let effects = pixel_effects(parts);
    for resolve in RESOLVES {
        if let Ok(source) = composite_source(resolve, &effects, None) {
            pipelines::add_generated(device, composite_path(resolve), PART.decl.brick, source);
        }
    }
}

fn shared_group<'c>(
    core: &'c crate::renderer::Core,
    layout: &'static crate::pipeline_layouts::PipelineLayout,
) -> Option<&'c wgpu::BindGroup> {
    match *layout.groups.first()? {
        GroupRef::Shared { group, without } => Some(core.shared_group(group, without)),
        GroupRef::Own(_) => None,
    }
}

impl GpuPart for PostPass {
    fn install(&mut self, table: &GraphTable) {
        self.ids = Ids {
            effects_pass: table.pass_named("effects"),
            composite_pass: table.pass_named("composite"),
            canvas: table.resource("canvas"),
            scene: table.resource("scene"),
            effects: table.resource("effects"),
            haze: table.resource("haze"),
        };
        self.linked = false;
    }

    fn begin(&mut self, cx: &mut Begin<'_>, _out: &mut BeginOut) {
        if !self.linked {
            self.linked = true;
            link_composites(&cx.core.device, cx.parts);
        }
    }

    fn prepare(&mut self, cx: &mut Step<'_>, _prelude: &mut Prelude) {
        if cx.frame.kind.debug {
            self.frame = None;
            return;
        }
        let snapshot = cx.frame.snapshot;
        let core = &*cx.core;
        let effects = crate::screen_effects::targets_read(&snapshot.environment);
        let region = core.display.region;
        let needs = cx.frame.temporal.is_some() || effects || region.upscales();
        let (device, queue) = (&core.device, &core.queue);
        let sky = cx
            .parts
            .get::<crate::shadows::ShadowPass>()
            .map(crate::shadows::ShadowPass::sky_bindings);
        self.frame = if needs {
            let format = core.canvas_format();
            self.post
                .create(device, "engine/post-", || Post::new(device, format));
            self.post.get_mut().map(|post| {
                post.begin(
                    device,
                    queue,
                    snapshot,
                    (region, &core.depth.view),
                    cx.frame.kind.display,
                    sky,
                )
            })
        } else {
            None
        };
        if !needs && let Some(post) = self.post.get_mut() {
            post.release_targets();
        }
    }

    fn finish(&mut self, cx: &mut Step<'_>) {
        let Some(frame) = self.frame else {
            return;
        };
        let Some(post) = self.post.get_mut() else {
            return;
        };
        let core = &*cx.core;
        let (Some(terrain), Some(scatter)) = (
            cx.parts.get::<crate::terrain::TerrainPass>(),
            cx.parts.get::<crate::scatter::ScatterPass>(),
        ) else {
            post.forget_reflections();
            return;
        };
        let snapshot = cx.frame.snapshot;
        let terrains = engine_terrain::RenderTerrains::of(snapshot);
        let infos: Vec<crate::scatter::TerrainInfo> = terrains
            .terrains
            .iter()
            .enumerate()
            .filter(|_| frame.reflections)
            .map(|(index, rendered)| crate::world_trace::terrain_info(terrain, index, rendered))
            .collect();
        let cover = engine_terrain::TerrainEnvironment::of(snapshot)
            .ground_cover
            .sanitized();
        let trace = frame.reflections.then(|| crate::world_trace::TraceFrame {
            terrains,
            infos: &infos,
            terrain,
            placed: scatter.placements(),
            cover: if cover.enabled { cover.density } else { 0.0 },
        });
        post.prepare_reflections(
            (&core.device, &core.queue),
            (frame, &core.depth.view),
            trace.as_ref(),
            (
                terrain.trace_sources(),
                scatter.slot_buffers(),
                core.impostors.placed.frames(),
            ),
        );
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let (Some(effects), Some(shape)) = (self.frame, cx.frame.shape) else {
            return;
        };
        let ids = self.ids;
        let targets = shape.targets;
        if targets != crate::graph::Targets::Canvas
            && effects.any()
            && let Some(pass) = ids.effects_pass
        {
            let _ = effects
                .reads(cx.table, graph.add(cx.table, pass))
                .write(ids.effects)
                .write_if(effects.haze, ids.haze);
        }
        if targets == crate::graph::Targets::Post
            && let (Some(pass), Some(canvas)) = (ids.composite_pass, ids.canvas)
        {
            let _ = graph
                .add(cx.table, pass)
                .read(ids.scene)
                .read_written(ids.effects)
                .color(canvas, Access::Clear(Clear::Color(BLACK)));
        }
    }

    fn encode(
        &mut self,
        pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        let (Some(post), Some(effects)) = (self.post.get(), self.frame) else {
            return 0;
        };
        if pass == COMPOSITE_PASS {
            let render = cx.render_pass(
                encoder,
                "engine/post-composite",
                cx.core.profiler.span("post_composite").map(Span::render),
            );
            post.draw_composite(render);
            return 0;
        }
        if pass == EFFECTS_PASS
            && let (Some(camera), Some(terrain_camera)) = (
                shared_group(cx.core, &HAZE_PIPELINES),
                shared_group(cx.core, &REFLECTION_PIPELINES),
            )
        {
            post.encode_effects(
                encoder,
                effects,
                (camera, terrain_camera),
                &cx.core.profiler,
            );
        }
        0
    }

    fn attachment(&self, resource: &str) -> Option<wgpu::TextureView> {
        let (scene, glow, surface) = self.post.get()?.targets()?;
        match resource {
            "scene" => Some(scene.clone()),
            "glow" => Some(glow.clone()),
            "surface" => Some(surface.clone()),
            _ => None,
        }
    }

    fn after_submit(&mut self) {
        let frame = self.frame;
        if let Some(post) = self.post.get_mut() {
            post.after_submit(frame.as_ref());
        }
    }

    fn capacity_bytes(&self) -> u64 {
        self.post.get().map_or(0, Post::capacity_bytes)
    }

    fn stats(&self, _cx: &View<'_>, stats: &mut RenderStats) {
        let applied = |effect: fn(&Effects) -> bool| Applied {
            active: self.frame.as_ref().is_some_and(effect),
        };
        stats.ambient_occlusion = applied(|effects| effects.occlusion);
        stats.bloom = applied(|effects| effects.bloom);
        stats.haze = applied(|effects| effects.haze);
        stats.reflections = applied(|effects| effects.reflections);
    }

    fn poll(&mut self) -> bool {
        self.post.poll()
    }

    fn disable(&mut self) -> bool {
        self.post.disable()
    }

    fn disabled(&self) -> bool {
        self.post.disabled()
    }

    #[cfg(feature = "dev")]
    fn inject(&mut self) -> bool {
        self.post.inject()
    }

    #[cfg(feature = "dev")]
    fn encode_fault(&self, encoder: &mut wgpu::CommandEncoder) {
        self.post.encode_fault(encoder);
    }
}

struct Targets {
    textures: [wgpu::Texture; 10],
    scene: wgpu::TextureView,
    glow: wgpu::TextureView,
    surface: wgpu::TextureView,
    scene_levels: wgpu::TextureView,
    scene_level_views: Vec<wgpu::TextureView>,
    scene_level_groups: Vec<wgpu::BindGroup>,
    occlusion: wgpu::TextureView,
    raw_occlusion: wgpu::TextureView,
    reflections: wgpu::TextureView,
    marched: wgpu::TextureView,
    haze: wgpu::TextureView,
    history: wgpu::TextureView,
    bloom: Vec<wgpu::TextureView>,
    bloom_first: wgpu::BindGroup,
    bloom_down: Vec<wgpu::BindGroup>,
    bloom_up: Vec<wgpu::BindGroup>,
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

type TraceInputs<'a> = (
    crate::terrain::TraceSources<'a>,
    (&'a wgpu::Buffer, &'a wgpu::Buffer, u64),
    Option<(&'a wgpu::TextureView, u64)>,
);

struct Post {
    layout: crate::pipelines::GroupLayout,
    fillers: Fillers,
    occlusion: wgpu::RenderPipeline,
    effects_filter: wgpu::RenderPipeline,
    reflections: wgpu::RenderPipeline,
    reflections_layout: crate::pipelines::GroupLayout,
    nearest: DepthPyramid,
    world_trace: crate::world_trace::WorldTrace,
    haze: wgpu::RenderPipeline,
    haze_layout: crate::pipelines::GroupLayout,
    haze_filter: wgpu::RenderPipeline,
    haze_history: Option<HazeHistory>,
    haze_kept: Option<HazeHistory>,
    haze_frames: u32,
    bloom_first: wgpu::RenderPipeline,
    bloom_down: wgpu::RenderPipeline,
    bloom_up: wgpu::RenderPipeline,
    scene_down: wgpu::RenderPipeline,
    bloom_format: wgpu::TextureFormat,
    levels_first: wgpu::RenderPipeline,
    levels_down: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    written: Option<PostUniform>,
    sampler: wgpu::Sampler,
    targets: ScreenTarget<Targets>,
    region: FrameRegion,
    occlusion_group: GroupCache<5>,
    effects_filter_group: GroupCache<2>,
    reflections_group: GroupCache<19>,
    haze_group: GroupCache<2>,
    haze_filter_group: GroupCache<4>,
}

#[derive(Clone, Copy)]
struct HazeHistory {
    view_projection: Mat4,
    eye: engine_core::glam::Vec3,
    drawn: (u32, u32),
}

impl Post {
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let bloom_format = bloom_format(device);
        let layout = pipelines::layout(device, &POST);
        let pipeline_layout = pipelines::layout_of(device, &POST_PIPELINES);
        let pipeline = |label, entry: crate::shaders::Entry, format: wgpu::TextureFormat, blend| {
            fullscreen(
                device,
                &Fullscreen {
                    label,
                    layout: &pipeline_layout,
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
                    vertex: post::vs_fullscreen,
                    fragment: entry,
                    constants: &[],
                    targets: &[Some(SCENE_FORMAT.into()), Some(bloom_format.into())],
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
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let haze_layout = pipelines::layout(device, &HAZE);
        let haze_pipeline_layout = pipelines::layout_of(device, &HAZE_PIPELINES);
        let march = fullscreen(
            device,
            &Fullscreen {
                label: "engine/post-haze",
                layout: &haze_pipeline_layout,
                vertex: haze::vs_fullscreen,
                fragment: haze::fs_haze,
                constants: &[],
                targets: &[Some(HAZE_FORMAT.into())],
                depth: None,
            },
        );
        let (reflections, reflections_layout) = reflections_pipeline(device);
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
            reflections,
            reflections_layout,
            nearest: DepthPyramid::new(device, Reduction::Nearest),
            world_trace: crate::world_trace::WorldTrace::new(device),
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
                SCENE_FORMAT,
                None,
            ),
            bloom_format,
            levels_first: levels_pipeline("engine/post-levels-first", post::fs_levels_first),
            levels_down: levels_pipeline("engine/post-levels-down", post::fs_levels_down),
            composite: pipeline(
                "engine/post-composite",
                composite_entry(PLAIN_RESOLVE),
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
            haze_group: GroupCache::default(),
            haze_filter_group: GroupCache::default(),
        }
    }

    fn release_targets(&mut self) {
        self.targets.release();
        self.occlusion_group.clear();
        self.effects_filter_group.clear();
        self.reflections_group.clear();
        self.haze_filter_group.clear();
        self.haze_history = None;
        self.nearest.release();
        self.world_trace.release();
    }

    fn forget_reflections(&mut self) {
        self.reflections_group.clear();
    }

    fn capacity_bytes(&self) -> u64 {
        self.targets.size().map_or(0, |size| {
            let pixels = u64::from(size.width) * u64::from(size.height);
            pixels * 48 + pixels * 4 / 3 + pixels / 4 * 8 * 4 / 3
        }) + self.nearest.capacity_bytes()
            + self.world_trace.capacity_bytes()
    }

    fn begin(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        snapshot: &RenderSnapshot,
        (region, depth): (FrameRegion, &Versioned<wgpu::TextureView>),
        display: bool,
        sky: Option<[(&'static str, Bound<'_>); 3]>,
    ) -> Effects {
        let size = region.display;
        if self.targets.resizes(size) {
            let targets = self.create_targets(device, size);
            self.targets.set(size, targets);
            self.haze_history = None;
        }
        self.region = region;
        let frame = Effects::of(&snapshot.environment);
        self.bind_occlusion(device, frame, depth, sky);
        let (history, turn) = self.begin_haze(device, (frame, depth), snapshot, display);
        if frame.reflections {
            self.nearest.prepare(device, queue, depth, region);
        }
        let (width, height) = (region.render.width as f32, region.render.height as f32);
        let (wide, tall) = (size.width as f32, size.height as f32);
        let view_projection = snapshot.view_projection;
        let focal = view_projection.row(1).truncate().length();
        let atmosphere = snapshot.environment.atmosphere.sanitized();
        let toward_sun = atmosphere.sun_direction.normalize_or_zero();
        let effects = frame.uniforms(
            &snapshot.environment,
            &UniformFrame {
                haze_turn: turn,
                haze_history: history
                    .map_or([0.0; 2], |last| [last.drawn.0 as f32, last.drawn.1 as f32]),
                world_terrains: crate::world_trace::WorldTrace::walked(
                    engine_terrain::RenderTerrains::of(snapshot),
                ),
                occupancy: engine_terrain::TerrainEnvironment::of(snapshot).occupancy,
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

    fn prepare_reflections(
        &mut self,
        (device, queue): (&wgpu::Device, &crate::gpu_calls::Queue),
        (frame, depth): (Effects, &Versioned<wgpu::TextureView>),
        trace: Option<&crate::world_trace::TraceFrame<'_>>,
        (sources, placed, frames): TraceInputs<'_>,
    ) {
        self.world_trace
            .prepare(device, queue, trace.filter(|_| frame.reflections));
        let (true, Some(targets), Some(nearest)) =
            (frame.reflections, self.targets.get(), self.nearest.view())
        else {
            return;
        };
        let world = self.world_trace.bindings(sources, placed, frames);
        let target = |view| Bound::of(targets, pipelines::view(view));
        let [
            heights,
            directory,
            control,
            palette,
            macro_maps,
            terrain_sky,
            terrains,
            placements,
            records,
            poses,
            groups,
            placed_frames,
        ] = world;
        self.reflections_group.bind(
            device,
            "engine/post-reflections-group",
            (&REFLECTIONS, &self.reflections_layout),
            [
                ("post", Bound::fixed(self.uniform.as_entire_binding())),
                ("depth_texture", depth.bound()),
                ("glow", target(&targets.glow)),
                ("linear", Bound::fixed(pipelines::sampler(&self.sampler))),
                ("nearest_depth", nearest.bound()),
                ("surface_texture", target(&targets.surface)),
                ("scene_levels", target(&targets.scene_levels)),
                heights,
                directory,
                control,
                palette,
                macro_maps,
                terrain_sky,
                terrains,
                placements,
                records,
                poses,
                groups,
                placed_frames,
            ],
        );
    }

    fn bind_occlusion(
        &mut self,
        device: &wgpu::Device,
        frame: Effects,
        depth: &Versioned<wgpu::TextureView>,
        sky: Option<[(&'static str, Bound<'_>); 3]>,
    ) {
        let targets = self.targets.get().expect("sized_in_begin");
        let post = || Bound::fixed(self.uniform.as_entire_binding());
        let target = |view| Bound::of(targets, pipelines::view(view));
        let shared = (&POST, &self.layout);
        if (frame.occlusion || frame.sky)
            && let Some([sky_maps, sky_sampler, sky]) = sky
        {
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
                    ("raw_occlusion", target(&targets.raw_occlusion)),
                ],
                &self.fillers,
            );
        }
    }

    fn begin_haze(
        &mut self,
        device: &wgpu::Device,
        (frame, depth): (Effects, &Versioned<wgpu::TextureView>),
        snapshot: &RenderSnapshot,
        display: bool,
    ) -> (Option<HazeHistory>, f32) {
        if frame.haze {
            let targets = self.targets.get().expect("sized_in_begin");
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
            drawn: self.region.render.pixels(),
        });
        if self.haze_kept.is_none() {
            return (None, 0.0);
        }
        let turn = (self.haze_frames % 1024) as f32 * 0.618_034;
        (self.haze_history, turn - turn.floor())
    }

    fn after_submit(&mut self, frame: Option<&Effects>) {
        let kept = self.haze_kept.take();
        self.haze_history = frame.and(kept);
        if self.haze_history.is_some() {
            self.haze_frames = self.haze_frames.wrapping_add(1);
        }
    }

    fn group(
        &self,
        device: &wgpu::Device,
        label: &str,
        resources: &[(&str, wgpu::BindingResource<'_>)],
    ) -> wgpu::BindGroup {
        filled_group(device, label, &POST, &self.layout, resources, &self.fillers)
    }

    fn targets(&self) -> Option<(&wgpu::TextureView, &wgpu::TextureView, &wgpu::TextureView)> {
        let targets = self.targets.get()?;
        Some((&targets.scene, &targets.glow, &targets.surface))
    }

    fn scene_texture(&self) -> Option<&wgpu::Texture> {
        Some(&self.targets.get()?.textures[0])
    }

    fn scene(&self) -> Option<Bound<'_>> {
        let targets = self.targets.get()?;
        Some(Bound::of(targets, pipelines::view(&targets.scene)))
    }

    fn resolve_inputs(&self) -> Option<([Bound<'_>; 5], Bound<'_>)> {
        let targets = self.targets.get()?;
        let target = |view| Bound::of(targets, pipelines::view(view));
        Some((
            [
                target(&targets.scene),
                target(&targets.occlusion),
                target(&targets.bloom[0]),
                target(&targets.reflections),
                target(&targets.haze),
            ],
            Bound::fixed(self.uniform.as_entire_binding()),
        ))
    }

    fn encode_effects(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame: Effects,
        (camera, terrain_camera): (&wgpu::BindGroup, &wgpu::BindGroup),
        profiler: &GpuProfiler,
    ) {
        let targets = self.targets.get().expect("sized_in_begin");
        let drawn = self.region.render.pixels();
        if (frame.occlusion || frame.sky)
            && let Some(group) = self.occlusion_group.get()
        {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-occlusion",
                (&targets.raw_occlusion, drawn),
                wgpu::LoadOp::Load,
                profiler.span("ambient_occlusion").map(Span::render),
            );
            pass.set_pipeline(&self.occlusion);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
        let reflecting = frame.reflections && self.reflections_group.get().is_some();
        let levels_done = if frame.bloom {
            self.encode_bloom(encoder, reflecting, profiler)
        } else {
            0
        };
        if reflecting && let Some(group) = self.reflections_group.get() {
            self.nearest.encode(encoder, profiler.span("reflections"));
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
                    (view, self.level_region(level as u32)),
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    None,
                );
                pass.set_pipeline(&self.scene_down);
                pass.set_bind_group(0, group, &[]);
                pass.draw(0..3, 0..1);
            }
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-reflections",
                (&targets.reflections, drawn),
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                profiler
                    .span("reflections")
                    .map(|span| span.render_part(false, true)),
            );
            pass.set_pipeline(&self.reflections);
            pass.set_bind_group(0, terrain_camera, &[]);
            pass.set_bind_group(1, group, &[]);
            pass.draw(0..3, 0..1);
        }
        if frame.haze {
            self.encode_haze(encoder, camera, profiler);
        }
        if (frame.occlusion || frame.sky)
            && let Some(group) = self.effects_filter_group.get()
        {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-effects-filter",
                (&targets.occlusion, drawn),
                wgpu::LoadOp::Load,
                profiler.span("effects_filter").map(Span::render),
            );
            pass.set_pipeline(&self.effects_filter);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

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
        let targets = self.targets.get().expect("sized_in_begin");
        let drawn = self.region.render.pixels();
        {
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-haze",
                (&targets.marched, drawn),
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
                (&targets.haze, drawn),
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
                width: drawn.0,
                height: drawn.1,
                depth_or_array_layers: 1,
            },
        );
    }

    fn encode_bloom(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene_levels: bool,
        profiler: &GpuProfiler,
    ) -> usize {
        let targets = self.targets.get().expect("sized_in_begin");
        let levels = targets.bloom.len();
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
        let drawn = |level: usize| self.level_region(level as u32);
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

    fn level_region(&self, level: u32) -> (u32, u32) {
        let FrameRegion { display, render } = self.region;
        let part = |drawn: u32, whole: u32| {
            let whole = (whole >> (level + 1)).max(1);
            ((drawn >> (level + 1)).max(1) + 1).min(whole)
        };
        (
            part(render.width, display.width),
            part(render.height, display.height),
        )
    }

    fn draw_composite(&self, mut pass: wgpu::RenderPass<'_>) {
        let targets = self.targets.get().expect("sized_in_begin");
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
        let size = (width, height);
        let half = ((width / 2).max(1), (height / 2).max(1));
        let texture = |label, size: (u32, u32), format, levels| {
            crate::targets::texture_2d(
                device,
                label,
                size,
                1,
                levels,
                format,
                crate::targets::RENDER_TARGET,
            )
        };
        let haze_texture = |label, usage| {
            crate::targets::texture_2d(device, label, size, 1, 1, HAZE_FORMAT, usage)
        };
        let levels = BLOOM_LEVELS.min(crate::gpu_data::mip_levels(half.0, half.1));
        let scene_levels = SCENE_LEVELS.min(crate::gpu_data::mip_levels(width, height) - 1);
        let textures = [
            crate::targets::texture_2d(
                device,
                "engine/post-scene",
                size,
                1,
                1 + scene_levels,
                SCENE_FORMAT,
                crate::targets::RENDER_TARGET | wgpu::TextureUsages::COPY_SRC,
            ),
            texture("engine/post-glow", size, GLOW_FORMAT, 1),
            texture("engine/post-occlusion", size, OCCLUSION_FORMAT, 1),
            texture("engine/post-bloom", half, self.bloom_format, levels),
            texture("engine/post-reflections", size, REFLECTION_FORMAT, 1),
            texture("engine/post-haze-march", size, HAZE_FORMAT, 1),
            texture("engine/post-occlusion-raw", size, RAW_OCCLUSION_FORMAT, 1),
            texture("engine/post-surface", size, SURFACE_FORMAT, 1),
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
            scene_levels_view,
            glow,
            occlusion,
            _,
            reflections,
            marched,
            raw_occlusion,
            surface,
            haze,
            history,
        ] = textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
        let level_view = |texture: &wgpu::Texture, level: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let scene = level_view(&textures[0], 0);
        let bloom: Vec<wgpu::TextureView> = (0..levels)
            .map(|level| level_view(&textures[3], level))
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
            .map(|level| {
                group(&[
                    uniform.clone(),
                    texture_entry("source", &bloom[level]),
                    sampler.clone(),
                ])
            })
            .collect();
        let bloom_up = (0..bloom.len().saturating_sub(1))
            .map(|level| {
                group(&[
                    uniform.clone(),
                    texture_entry("source", &bloom[level + 1]),
                    sampler.clone(),
                ])
            })
            .collect();
        let scene_level_views: Vec<wgpu::TextureView> = (1..=scene_levels)
            .map(|level| level_view(&textures[0], level))
            .collect();
        let scene_level_groups = (0..scene_level_views.len())
            .map(|level| {
                let source = if level == 0 {
                    &scene
                } else {
                    &scene_level_views[level - 1]
                };
                group(&[
                    uniform.clone(),
                    texture_entry("source", source),
                    sampler.clone(),
                ])
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
                    uniform.clone(),
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
            ("bloom_sampler", pipelines::sampler(&self.sampler)),
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

fn reflections_pipeline(
    device: &wgpu::Device,
) -> (wgpu::RenderPipeline, crate::pipelines::GroupLayout) {
    let layout = pipelines::layout(device, &REFLECTIONS);
    let pipeline_layout = pipelines::layout_of(device, &REFLECTION_PIPELINES);
    let pipeline = fullscreen(
        device,
        &Fullscreen {
            label: "engine/post-reflections",
            layout: &pipeline_layout,
            vertex: reflections::vs_fullscreen,
            fragment: reflections::fs_reflections,
            constants: &[],
            targets: &[Some(REFLECTION_FORMAT.into())],
            depth: None,
        },
    );
    (pipeline, layout)
}

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
