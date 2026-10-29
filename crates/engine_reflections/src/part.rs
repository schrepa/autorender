use engine_core::{Applied, RenderStats};
use engine_faces::heightfield::HeightfieldFace;
use engine_faces::placed::PlacedFace;
use engine_render::bindings::GroupRef;
use engine_render::cached::{Bound, GroupCache, Versioned};
use engine_render::depth_pyramid::{DepthPyramid, Reduction};
use engine_render::gpu_calls::Queue;
use engine_render::graph::{
    Access, Clear, FrameGraph, GraphTable, PassDecl, PassId, ResourceId, Stage, TRANSPARENT,
    Targets,
};
use engine_render::lazy::Lazy;
use engine_render::parts::{Create, Encode, GpuPart, PartDecl, PartEntry, Prelude, Step, View};
use engine_render::pieces::PieceDecl;
use engine_render::pipeline_layouts::PipelineLayout;
use engine_render::pipelines::{self, Fullscreen, GroupLayout, fullscreen};
use engine_render::post::PostPass;
use engine_render::profiling::GpuProfiler;
use engine_render::renderer::Core;
use engine_render::shaders::Entry;
use engine_render::targets::{FrameRegion, FrameSize, RENDER_TARGET, ScreenTarget, draw_within};
use engine_terrain::{RenderTerrains, TerrainEnvironment};

use crate::gpu_layouts::{ReflectionsPiece, TerrainInfo};
use crate::layouts::{
    REFLECTION_LEVEL_PIPELINES, REFLECTION_LEVELS, REFLECTION_PIPELINES, REFLECTIONS,
};
use crate::shaders::entry::reflections;
use crate::world_trace::{Placements, TraceFrame, WorldTrace, terrain_info};

const SCENE_LEVELS: u32 = 6;

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "reflections",
        brick: "reflections",
        passes: &[PassDecl {
            name: "reflections",
            stage: Stage::PostScene,
        }],
        resources: &[],
        hooks: &[],
        contributes: &[],
        effects: &[],
        labels: &["engine/reflections-"],
        lazy: true,
        faces: &[],
        needs: &[],
        piece: Some(PieceDecl {
            name: "reflections",
            module: "reflections::piece",
            ty: "ReflectionsPiece",
            size: size_of::<ReflectionsPiece>() as u64,
        }),
    },
    create,
};

fn create(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(ReflectionsPass::default())
}

#[derive(Clone, Copy, Default)]
struct Ids {
    pass: Option<PassId>,
    depth: Option<ResourceId>,
    scene: Option<ResourceId>,
    glow: Option<ResourceId>,
    surface: Option<ResourceId>,
    reflections: Option<ResourceId>,
}

#[derive(Default)]
struct ReflectionsPass {
    reflections: Lazy<Reflections>,
    active: bool,
    piece: ReflectionsPiece,
    ids: Ids,
}

fn shared_group<'c>(
    core: &'c Core,
    layout: &'static PipelineLayout,
) -> Option<&'c wgpu::BindGroup> {
    match *layout.groups.first()? {
        GroupRef::Shared { group, without } => Some(core.shared_group(group, without)),
        GroupRef::Own(_) => None,
    }
}

impl GpuPart for ReflectionsPass {
    fn install(&mut self, table: &GraphTable) {
        self.ids = Ids {
            pass: table.pass_named("reflections"),
            depth: table.resource("depth"),
            scene: table.resource("scene"),
            glow: table.resource("glow"),
            surface: table.resource("surface"),
            reflections: table.resource("reflections"),
        };
    }

    fn prepare(&mut self, cx: &mut Step<'_>, _prelude: &mut Prelude) {
        let snapshot = cx.frame.snapshot;
        let settings = snapshot.environment.reflections.sanitized();
        self.piece = ReflectionsPiece {
            surface: [
                settings.max_roughness,
                if settings.enabled() { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
            trace: [
                settings.distance,
                0.0,
                0.0,
                WorldTrace::walked(RenderTerrains::of(snapshot)) as f32,
            ],
        };
        self.active = false;
        if cx.frame.kind.debug {
            return;
        }
        let core = &*cx.core;
        let Some(post) = cx.parts.post().filter(|post| post.active()) else {
            if let Some(reflections) = self.reflections.get_mut() {
                reflections.release_targets();
            }
            return;
        };
        if !settings.enabled() {
            if let Some(reflections) = self.reflections.get_mut() {
                reflections.world_trace.release();
            }
            return;
        }
        let target = post.attachment("reflections");
        let (Some(scene), Some(target)) = (post.scene_texture(), target) else {
            return;
        };
        let formats = (target.texture().format(), scene.format());
        let device = &core.device;
        self.reflections
            .create(device, "engine/reflections-", || Reflections::new(device, formats));
        if let Some(reflections) = self.reflections.get_mut() {
            self.active = reflections.begin(
                device,
                &core.queue,
                (core.display.region, &core.depth.view),
                post,
            );
        }
    }

    fn finish(&mut self, cx: &mut Step<'_>) {
        if !self.active {
            return;
        }
        let Some(reflections) = self.reflections.get_mut() else {
            return;
        };
        let core = &*cx.core;
        let (Some(terrain), Some(placed), Some((_, uniform))) = (
            cx.parts.face::<dyn HeightfieldFace>(),
            cx.parts.face::<dyn PlacedFace>(),
            cx.parts.post().and_then(|post| post.resolve_inputs()),
        ) else {
            reflections.forget();
            return;
        };
        let snapshot = cx.frame.snapshot;
        let terrains = RenderTerrains::of(snapshot);
        let infos: Vec<TerrainInfo> = terrains
            .terrains
            .iter()
            .enumerate()
            .map(|(index, rendered)| terrain_info(terrain, index, rendered))
            .collect();
        let cover = TerrainEnvironment::of(snapshot).ground_cover.sanitized();
        let trace = TraceFrame {
            terrains,
            infos: &infos,
            terrain,
            placed: Placements::of(placed.placements()),
            cover: if cover.enabled { cover.density } else { 0.0 },
        };
        reflections.prepare_reflections(
            (&core.device, &core.queue),
            (&core.depth.view, uniform),
            &trace,
            (
                terrain.trace_sources(),
                placed.slot_buffers(),
                placed.frames(),
            ),
        );
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let (true, Some(shape), Some(pass), Some(reflections)) = (
            self.active,
            cx.frame.shape,
            self.ids.pass,
            self.ids.reflections,
        ) else {
            return;
        };
        if shape.targets == Targets::Canvas {
            return;
        }
        let ids = self.ids;
        let _ = graph
            .add(cx.table, pass)
            .read(ids.depth)
            .read(ids.scene)
            .read(ids.glow)
            .read(ids.surface)
            .color(reflections, Access::Clear(Clear::Color(TRANSPARENT)));
    }

    fn encode(
        &mut self,
        _pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        let Some(reflections) = self.reflections.get().filter(|_| self.active) else {
            return 0;
        };
        let (Some(frame), Some(group), Some(scene)) = (
            shared_group(cx.core, &REFLECTION_PIPELINES),
            reflections.group.get(),
            cx.parts.post().and_then(|post| post.scene_texture()),
        ) else {
            return 0;
        };
        let profiler = &cx.core.profiler;
        reflections.encode_levels(encoder, scene, profiler);
        let mut pass = cx.render_pass(
            encoder,
            "engine/post-reflections",
            profiler
                .span("reflections")
                .map(|span| span.render_part(false, true)),
        );
        pass.set_pipeline(&reflections.pipeline);
        pass.set_bind_group(0, frame, &[]);
        pass.set_bind_group(1, group, &[]);
        pass.draw(0..3, 0..1);
        0
    }

    fn capacity_bytes(&self) -> u64 {
        self.reflections
            .get()
            .map_or(0, Reflections::capacity_bytes)
    }

    fn stats(&self, _cx: &View<'_>, stats: &mut RenderStats) {
        stats.reflections = Applied {
            active: self.active,
        };
    }

    fn poll(&mut self) -> bool {
        self.reflections.poll()
    }

    fn disable(&mut self) -> bool {
        self.reflections.disable()
    }

    fn disabled(&self) -> bool {
        self.reflections.disabled()
    }

    #[cfg(feature = "dev")]
    fn inject(&mut self) -> bool {
        self.reflections.inject()
    }

    #[cfg(feature = "dev")]
    fn encode_fault(&self, encoder: &mut wgpu::CommandEncoder) {
        self.reflections.encode_fault(encoder);
    }

    fn piece(&self) -> Option<&[u8]> {
        Some(bytemuck::bytes_of(&self.piece))
    }

    fn produces(&self, resource: ResourceId) -> bool {
        self.active && self.ids.reflections == Some(resource)
    }
}

#[derive(Default)]
struct Watched {
    views: Option<[wgpu::TextureView; 2]>,
    generation: u64,
}

impl Watched {
    fn watch(&mut self, views: [wgpu::TextureView; 2]) {
        if self.views.as_ref() != Some(&views) {
            self.views = Some(views);
            self.generation += 1;
        }
    }
}

struct LevelTargets {
    texture: wgpu::Texture,
    scene: wgpu::Texture,
    levels: wgpu::TextureView,
    views: Vec<wgpu::TextureView>,
    bytes: u64,
}

impl Drop for LevelTargets {
    fn drop(&mut self) {
        self.texture.destroy();
    }
}

impl LevelTargets {
    fn new(
        device: &wgpu::Device,
        FrameSize { width, height }: FrameSize,
        scene: &wgpu::Texture,
    ) -> Self {
        let count = SCENE_LEVELS.min(engine_render::gpu_data::mip_levels(width, height) - 1);
        let format = scene.format();
        let texture = engine_render::targets::texture_2d(
            device,
            "engine/post-scene-levels",
            (width, height),
            1,
            1 + count,
            format,
            RENDER_TARGET | wgpu::TextureUsages::COPY_DST,
        );
        let level_view = |level: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let views = (0..=count).map(level_view).collect();
        let texel = u64::from(format.block_copy_size(None).unwrap_or(4));
        let bytes = (0..=count)
            .map(|level| {
                u64::from((width >> level).max(1)) * u64::from((height >> level).max(1)) * texel
            })
            .sum();
        Self {
            levels: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            texture,
            scene: scene.clone(),
            views,
            bytes,
        }
    }
}

type TraceInputs<'a> = (
    engine_faces::heightfield::TraceSources<'a>,
    (&'a wgpu::Buffer, &'a wgpu::Buffer, u64),
    Option<(&'a wgpu::TextureView, u64)>,
);

struct Reflections {
    pipeline: wgpu::RenderPipeline,
    layout: GroupLayout,
    levels_first: wgpu::RenderPipeline,
    levels_down: wgpu::RenderPipeline,
    levels_layout: GroupLayout,
    sampler: wgpu::Sampler,
    nearest: DepthPyramid,
    world_trace: WorldTrace,
    targets: ScreenTarget<LevelTargets>,
    region: FrameRegion,
    inputs: Watched,
    level_groups: Vec<GroupCache<3>>,
    group: GroupCache<19>,
}

impl Reflections {
    fn new(
        device: &wgpu::Device,
        (target, scene): (wgpu::TextureFormat, wgpu::TextureFormat),
    ) -> Self {
        let pipeline_layout = pipelines::layout_of(device, &REFLECTION_PIPELINES);
        let pipeline = fullscreen(
            device,
            &Fullscreen {
                label: "engine/post-reflections",
                layout: &pipeline_layout,
                vertex: reflections::vs_fullscreen,
                fragment: reflections::fs_reflections,
                constants: &[],
                targets: &[Some(target.into())],
                depth: None,
            },
        );
        let levels_pipeline_layout = pipelines::layout_of(device, &REFLECTION_LEVEL_PIPELINES);
        let level = |label, entry: Entry| {
            fullscreen(
                device,
                &Fullscreen {
                    label,
                    layout: &levels_pipeline_layout,
                    vertex: reflections::vs_fullscreen,
                    fragment: entry,
                    constants: &[],
                    targets: &[Some(wgpu::ColorTargetState {
                        format: scene,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    depth: None,
                },
            )
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/post-reflections-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            layout: pipelines::layout(device, &REFLECTIONS),
            levels_first: level("engine/post-levels-first", reflections::fs_levels_first),
            levels_down: level("engine/post-scene-down", reflections::fs_levels_down),
            levels_layout: pipelines::layout(device, &REFLECTION_LEVELS),
            sampler,
            nearest: DepthPyramid::new(device, Reduction::Nearest),
            world_trace: WorldTrace::new(device),
            targets: ScreenTarget::default(),
            region: FrameRegion::default(),
            inputs: Watched::default(),
            level_groups: Vec::new(),
            group: GroupCache::default(),
        }
    }

    fn release_targets(&mut self) {
        self.targets.release();
        self.level_groups.clear();
        self.group.clear();
        self.inputs.views = None;
        self.nearest.release();
        self.world_trace.release();
    }

    fn forget(&mut self) {
        self.group.clear();
    }

    fn capacity_bytes(&self) -> u64 {
        self.targets.get().map_or(0, |targets| targets.bytes)
            + self.nearest.capacity_bytes()
            + self.world_trace.capacity_bytes()
    }

    fn begin(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        (region, depth): (FrameRegion, &Versioned<wgpu::TextureView>),
        post: &PostPass,
    ) -> bool {
        let (Some(scene), Some(glow), Some(surface), Some((_, uniform))) = (
            post.scene_texture(),
            post.attachment("glow"),
            post.attachment("surface"),
            post.resolve_inputs(),
        ) else {
            return false;
        };
        self.region = region;
        self.inputs.watch([glow, surface]);
        let size = region.display;
        if self
            .targets
            .get()
            .is_some_and(|targets| targets.scene != *scene)
        {
            self.targets.release();
        }
        if self.targets.resizes(size) {
            let targets = LevelTargets::new(device, size, scene);
            self.targets.set(size, targets);
        }
        let Some(targets) = self.targets.get() else {
            return false;
        };
        self.level_groups
            .resize_with(targets.views.len() - 1, GroupCache::default);
        for (level, cache) in self.level_groups.iter_mut().enumerate() {
            cache.bind(
                device,
                "engine/post-group",
                (&REFLECTION_LEVELS, &self.levels_layout),
                [
                    ("post", uniform.clone()),
                    (
                        "source",
                        Bound::of(targets, pipelines::view(&targets.views[level])),
                    ),
                    ("linear", Bound::fixed(pipelines::sampler(&self.sampler))),
                ],
            );
        }
        self.nearest.prepare(device, queue, depth, region);
        true
    }

    fn prepare_reflections(
        &mut self,
        (device, queue): (&wgpu::Device, &Queue),
        (depth, uniform): (&Versioned<wgpu::TextureView>, Bound<'_>),
        trace: &TraceFrame<'_>,
        (sources, placed, frames): TraceInputs<'_>,
    ) {
        self.world_trace.prepare(device, queue, Some(trace));
        let (Some(targets), Some(nearest), Some([glow, surface])) = (
            self.targets.get(),
            self.nearest.view(),
            self.inputs.views.as_ref(),
        ) else {
            return;
        };
        let watched = self.inputs.generation;
        let world = self.world_trace.bindings(sources, placed, frames);
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
        self.group.bind(
            device,
            "engine/post-reflections-group",
            (&REFLECTIONS, &self.layout),
            [
                ("post", uniform),
                ("depth_texture", depth.bound()),
                ("glow", Bound::counted(watched, pipelines::view(glow))),
                ("linear", Bound::fixed(pipelines::sampler(&self.sampler))),
                ("nearest_depth", nearest.bound()),
                (
                    "surface_texture",
                    Bound::counted(watched, pipelines::view(surface)),
                ),
                (
                    "scene_levels",
                    Bound::of(targets, pipelines::view(&targets.levels)),
                ),
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

    fn encode_levels(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &wgpu::Texture,
        profiler: &GpuProfiler,
    ) {
        self.nearest.encode(encoder, profiler.span("reflections"));
        let Some(targets) = self.targets.get() else {
            return;
        };
        if targets.scene != *scene {
            return;
        }
        encoder.copy_texture_to_texture(
            scene.as_image_copy(),
            targets.texture.as_image_copy(),
            wgpu::Extent3d {
                width: targets.texture.width(),
                height: targets.texture.height(),
                depth_or_array_layers: 1,
            },
        );
        for (level, (view, group)) in targets
            .views
            .iter()
            .skip(1)
            .zip(&self.level_groups)
            .enumerate()
        {
            let Some(group) = group.get() else {
                continue;
            };
            let mut pass = fullscreen_pass(
                encoder,
                "engine/post-scene-levels",
                (view, self.level_region(level as u32)),
                wgpu::LoadOp::Load,
                None,
            );
            pass.set_pipeline(if level == 0 {
                &self.levels_first
            } else {
                &self.levels_down
            });
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
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
