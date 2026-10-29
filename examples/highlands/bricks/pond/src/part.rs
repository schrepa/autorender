use bytemuck::Zeroable;
use engine_faces::heightfield::{CoverSources, HeightfieldFace};
use engine_render::graph::{
    Access, FrameGraph, GraphTable, PassDecl, PassId, ResourceId, Stage, Targets,
};
use engine_render::parts::{Create, Encode, GpuPart, PartDecl, PartEntry, Step, View, main_targets};
use engine_render::pipelines::{self, Fragment, Geometry, GroupLayout};
use engine_render::profiling::Span;
use engine_terrain::{RenderTerrain, RenderTerrains};
use glam::Vec2;

use crate::layouts::{POND, POND_MOST, POND_PIPELINES, PondUniform};
use crate::shaders::entry::pond;
use crate::{Pond, RenderPonds};

pub static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "pond",
        brick: "pond",
        passes: &[PassDecl {
            name: "pond",
            stage: Stage::Surfaces,
        }],
        resources: &[],
        hooks: &[],
        contributes: &[],
        effects: &[],
        labels: &[],
        lazy: false,
        faces: &[],
        needs: &[],
        piece: None,
    },
    create,
};

fn create(cx: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(PondPass::new(&cx.core.device))
}

#[derive(Clone, Copy, Default)]
struct Ids {
    pass: Option<PassId>,
    depth: Option<ResourceId>,
    shadow_maps: Option<ResourceId>,
    probes: Option<ResourceId>,
}

struct PondPass {
    layout: GroupLayout,
    uniform: wgpu::Buffer,
    group: Option<((u64, u64), wgpu::BindGroup)>,
    pipelines: Vec<(Targets, wgpu::RenderPipeline)>,
    drawn: u32,
    ids: Ids,
}

impl PondPass {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            layout: pipelines::layout(device, &POND),
            uniform: engine_render::buffers::uniform(
                device,
                "pond/ponds",
                size_of::<PondUniform>() as u64,
            ),
            group: None,
            pipelines: Vec::new(),
            drawn: 0,
            ids: Ids::default(),
        }
    }

    fn pipeline(&self, targets: Targets) -> Option<&wgpu::RenderPipeline> {
        self.pipelines
            .iter()
            .find(|(known, _)| *known == targets)
            .map(|(_, pipeline)| pipeline)
    }

    fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        targets: Targets,
        canvas: wgpu::TextureFormat,
    ) {
        if self.pipeline(targets).is_some() {
            return;
        }
        let fragment = match targets {
            Targets::Canvas => pond::fs_pond,
            Targets::Post => pond::fs_pond_post,
            Targets::Temporal => pond::fs_pond_temporal,
            Targets::Debug => return,
        };
        let layout = pipelines::layout_of(device, &POND_PIPELINES);
        let covering = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let mut color_targets = targets.color_targets(canvas);
        if let Some(Some(scene)) = color_targets.first_mut() {
            scene.blend = Some(covering);
        }
        let pipeline = pipelines::geometry(
            device,
            &Geometry {
                label: "pond/surface",
                layout: &layout,
                vertex: pond::vs_pond,
                vertex_constants: &[],
                buffers: &[],
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth: Some(wgpu::DepthStencilState {
                    format: engine_render::renderer::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                fragment: Some(Fragment {
                    entry: fragment,
                    constants: &[],
                    targets: &color_targets,
                }),
            },
        );
        self.pipelines.push((targets, pipeline));
    }

    fn bind(&mut self, device: &wgpu::Device, sources: &CoverSources<'_>) {
        let generation = (sources.generation.0, sources.generation.1);
        if self
            .group
            .as_ref()
            .is_some_and(|(bound, _)| *bound == generation)
        {
            return;
        }
        let group = pipelines::group(
            device,
            "pond/group",
            &POND,
            &self.layout,
            &[
                ("ponds", self.uniform.as_entire_binding()),
                ("heights", pipelines::view(sources.heights)),
                ("directory", sources.directory.as_entire_binding()),
            ],
        );
        self.group = Some((generation, group));
    }
}

fn shapes(ponds: &[Pond], terrains: &[RenderTerrain], ground: &dyn HeightfieldFace) -> PondUniform {
    let mut uniform = PondUniform::zeroed();
    for (shape, pond) in uniform.shapes.iter_mut().zip(ponds) {
        let (low, high) = (pond.low(), pond.high());
        shape.area = [low.x, low.y, high.x, high.y];
        shape.surface = [pond.surface, 0.0, 0.0, 0.0];
        if let Some((index, terrain)) = under(terrains, pond) {
            let origin = terrain.origin;
            shape.terrain = [origin.x, origin.y, origin.z, terrain.pyramid.spacing(0)];
            let (directory, _, _) = ground.cover_terrain(index);
            shape.ground = [directory, terrain.pyramid.levels(), 0, 0];
        }
    }
    uniform.count = [ponds.len().min(POND_MOST) as u32, 0, 0, 0];
    uniform
}

fn under<'t>(terrains: &'t [RenderTerrain], pond: &Pond) -> Option<(usize, &'t RenderTerrain)> {
    terrains.iter().enumerate().find(|(_, terrain)| {
        let local = pond.center - Vec2::new(terrain.origin.x, terrain.origin.z);
        let extent = terrain.pyramid.extent();
        !terrain.surroundings
            && local.x >= 0.0
            && local.y >= 0.0
            && local.x <= extent.x
            && local.y <= extent.y
    })
}

impl GpuPart for PondPass {
    fn install(&mut self, table: &GraphTable) {
        self.ids = Ids {
            pass: table.pass_named("pond"),
            depth: table.resource("depth"),
            shadow_maps: table.resource("shadow_maps"),
            probes: table.resource("probes"),
        };
        self.pipelines.clear();
        self.group = None;
    }

    fn forget_targets(&mut self, targets: Targets) {
        self.pipelines.retain(|(known, _)| *known != targets);
    }

    fn finish(&mut self, cx: &mut Step<'_>) {
        self.drawn = 0;
        let Some(shape) = cx.frame.shape else {
            return;
        };
        let ponds = RenderPonds::of(cx.frame.snapshot);
        if ponds.is_empty() || shape.targets == Targets::Debug {
            return;
        }
        let Some(ground) = cx.parts.face::<dyn HeightfieldFace>() else {
            return;
        };
        let Some(sources) = ground.cover_sources() else {
            return;
        };
        let terrains = &RenderTerrains::of(cx.frame.snapshot).terrains;
        let uniform = shapes(ponds, terrains, ground);
        let core = &*cx.core;
        core.queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        self.bind(&core.device, &sources);
        self.ensure_pipeline(&core.device, shape.targets, core.display.format());
        self.drawn = uniform.count[0];
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let (Some(pass), Some(shape)) = (self.ids.pass, cx.frame.shape) else {
            return;
        };
        if self.drawn == 0 || self.group.is_none() || self.pipeline(shape.targets).is_none() {
            return;
        }
        let mut pond = graph
            .add(cx.table, pass)
            .read_written(self.ids.shadow_maps)
            .read(self.ids.probes);
        for (target, _) in main_targets(cx.table, &shape) {
            pond = pond.color(target, Access::Load);
        }
        if let Some(depth) = self.ids.depth {
            let _ = pond.depth(depth, Access::Load);
        }
    }

    fn encode(
        &mut self,
        _pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        let (Some(shape), Some((_, group))) = (cx.frame.shape, &self.group) else {
            return 0;
        };
        let Some(pipeline) = self.pipeline(shape.targets) else {
            return 0;
        };
        let core = &*cx.core;
        let mut pass = cx.render_pass(
            encoder,
            "pond/surface",
            core.profiler.span("pond").map(Span::render),
        );
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, core.shared_group("frame", &["refraction"]), &[]);
        pass.set_bind_group(1, group, &[]);
        pass.draw(0..6, 0..self.drawn);
        1
    }

    fn capacity_bytes(&self) -> u64 {
        self.uniform.size()
    }
}
