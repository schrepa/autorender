#![expect(
    clippy::expect_used,
    reason = "the pass binds the targets its frame made"
)]

use engine_core::{DebugView, DebugViewMode};
use engine_render::graph::{
    Access, Clear, FrameGraph, GraphTable, PassDecl, PassId, ResourceId, Stage,
};
use engine_render::lazy::Lazy;
use engine_render::parts::{
    Begin, BeginOut, Create, Encode, GpuPart, PartDecl, PartEntry, Step, View,
};

use crate::gpu_layouts::{
    DEBUG_VIEW_CARDS, DEBUG_VIEW_CASTERS, DEBUG_VIEW_DETAIL, DEBUG_VIEW_KINDS, DEBUG_VIEW_LEVELS,
    DEBUG_VIEW_MATERIALS, DEBUG_VIEW_NORMALS, DEBUG_VIEW_OVERDRAW, DEBUG_VIEW_SHADOWS,
    DEBUG_VIEW_SKY, DEBUG_VIEW_TILES, DEBUG_VIEW_WATER_DEPTH, DEBUG_VIEW_WATER_EDDIES,
    DEBUG_VIEW_WATER_FROUDE, DEBUG_VIEW_WATER_REACH, DEBUG_VIEW_WATER_ROUGHNESS,
    DEBUG_VIEW_WATER_SPAN, DEBUG_VIEW_WATER_SPEED, DEBUG_VIEW_WATER_TURBULENCE,
    DEBUG_VIEW_WATER_WHITE, DebugViewUniform,
};
use crate::pipeline_layouts::{DEBUG_VIEW, DEBUG_VIEW_PIPELINES};
use crate::shaders::entry::debug_view;

fn view_value(view: DebugViewMode) -> u32 {
    match view {
        DebugViewMode::Off => 0,
        DebugViewMode::Kinds => DEBUG_VIEW_KINDS,
        DebugViewMode::Levels => DEBUG_VIEW_LEVELS,
        DebugViewMode::Cards => DEBUG_VIEW_CARDS,
        DebugViewMode::Shadows => DEBUG_VIEW_SHADOWS,
        DebugViewMode::Casters => DEBUG_VIEW_CASTERS,
        DebugViewMode::Overdraw => DEBUG_VIEW_OVERDRAW,
        DebugViewMode::Normals => DEBUG_VIEW_NORMALS,
        DebugViewMode::Tiles => DEBUG_VIEW_TILES,
        DebugViewMode::Materials => DEBUG_VIEW_MATERIALS,
        DebugViewMode::Sky => DEBUG_VIEW_SKY,
        DebugViewMode::Detail => DEBUG_VIEW_DETAIL,
        DebugViewMode::WaterDepth => DEBUG_VIEW_WATER_DEPTH,
        DebugViewMode::WaterSpeed => DEBUG_VIEW_WATER_SPEED,
        DebugViewMode::WaterFroude => DEBUG_VIEW_WATER_FROUDE,
        DebugViewMode::WaterTurbulence => DEBUG_VIEW_WATER_TURBULENCE,
        DebugViewMode::WaterWhite => DEBUG_VIEW_WATER_WHITE,
        DebugViewMode::WaterReach => DEBUG_VIEW_WATER_REACH,
        DebugViewMode::WaterSpan => DEBUG_VIEW_WATER_SPAN,
        DebugViewMode::WaterEddies => DEBUG_VIEW_WATER_EDDIES,
        DebugViewMode::WaterRoughness => DEBUG_VIEW_WATER_ROUGHNESS,
    }
}

pub static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "debug_view",
        brick: "debug_view",
        passes: &[PassDecl {
            name: "debug_view",
            stage: Stage::Composite,
        }],
        resources: &[],
        hooks: &[],
        contributes: &[],
        effects: &[],
        labels: &["engine/debug-view"],
        lazy: true,
        faces: &[],
        needs: &[],
        piece: None,
    },
    create,
};

fn create(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(DebugViewPass::default())
}

#[derive(Clone, Copy, Default)]
struct Ids {
    pass: Option<PassId>,
    canvas: Option<ResourceId>,
    depth: Option<ResourceId>,
    ids: Option<ResourceId>,
    count: Option<ResourceId>,
    terrain: Option<ResourceId>,
}

#[derive(Default)]
struct DebugViewPass {
    view: Lazy<DebugTargets>,
    frame: Option<DebugViewFrame>,
    shows: bool,
    ids: Ids,
}

struct DebugTargets {
    layout: engine_render::pipelines::GroupLayout,
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    group: Option<([wgpu::TextureView; 3], wgpu::BindGroup)>,
}

#[derive(Clone, Copy)]
struct DebugViewFrame;

impl DebugTargets {
    fn new(device: &wgpu::Device, canvas: wgpu::TextureFormat) -> Self {
        let layout = engine_render::pipelines::layout(device, &DEBUG_VIEW);
        let pipeline_layout = engine_render::pipelines::layout_of(device, &DEBUG_VIEW_PIPELINES);
        let pipeline = engine_render::pipelines::fullscreen(
            device,
            &engine_render::pipelines::Fullscreen {
                label: "engine/debug-view",
                layout: &pipeline_layout,
                vertex: debug_view::vs_fullscreen,
                fragment: debug_view::fs_debug_view,
                constants: &[],
                targets: &[Some(canvas.into())],
                depth: None,
            },
        );
        Self {
            layout,
            pipeline,
            uniform: engine_render::buffers::uniform(
                device,
                "engine/debug-view-uniform",
                size_of::<DebugViewUniform>() as u64,
            ),
            group: None,
        }
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
        targets: [wgpu::TextureView; 3],
        settings: DebugView,
        sun: [f32; 3],
    ) -> DebugViewFrame {
        self.bind(device, targets);
        let view = if settings.showing() {
            settings.view
        } else {
            DebugViewMode::Kinds
        };
        let uniform = DebugViewUniform {
            view: [view_value(view), u32::from(settings.shade), 0, 0],
            sun: [sun[0], sun[1], sun[2], 0.0],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        DebugViewFrame
    }

    fn bind(&mut self, device: &wgpu::Device, targets: [wgpu::TextureView; 3]) {
        if self
            .group
            .as_ref()
            .is_some_and(|(bound, _)| *bound == targets)
        {
            return;
        }
        let [ids, count, terrain] = &targets;
        let group = engine_render::pipelines::group(
            device,
            "engine/debug-view-group",
            &DEBUG_VIEW,
            &self.layout,
            &[
                (
                    "debug_view",
                    wgpu::BindingResource::Buffer(self.uniform.as_entire_buffer_binding()),
                ),
                ("debug_ids", engine_render::pipelines::view(ids)),
                ("debug_count", engine_render::pipelines::view(count)),
                ("debug_terrain", engine_render::pipelines::view(terrain)),
            ],
        );
        self.group = Some((targets, group));
    }
}

impl GpuPart for DebugViewPass {
    fn install(&mut self, table: &GraphTable) {
        self.ids = Ids {
            pass: table.pass_named("debug_view"),
            canvas: table.resource("canvas"),
            depth: table.resource("depth"),
            ids: table.resource("debug_ids"),
            count: table.resource("debug_count"),
            terrain: table.resource("debug_terrain"),
        };
    }

    fn begin(&mut self, cx: &mut Begin<'_>, _out: &mut BeginOut) {
        self.shows = cx.snapshot.environment.debug_view.showing() && !self.view.disabled();
    }

    fn wants(&self, resource: ResourceId) -> bool {
        self.shows && [self.ids.ids, self.ids.count, self.ids.terrain].contains(&Some(resource))
    }

    fn finish(&mut self, cx: &mut Step<'_>) {
        if !cx.frame.kind.debug {
            self.frame = None;
            return;
        }
        let snapshot = cx.frame.snapshot;
        let core = &*cx.core;
        let (device, queue) = (&core.device, &core.queue);
        let format = core.canvas_format();
        let targets = cx.parts.post().and_then(|post| {
            Some([
                post.attachment("debug_ids")?,
                post.attachment("debug_count")?,
                post.attachment("debug_terrain")?,
            ])
        });
        let sun = snapshot
            .environment
            .atmosphere
            .sun_direction
            .normalize_or_zero()
            .to_array();
        self.view.create(device, "engine/debug-view", || {
            DebugTargets::new(device, format)
        });
        self.frame = self.view.get_mut().zip(targets).map(|(view, targets)| {
            view.prepare(device, queue, targets, snapshot.environment.debug_view, sun)
        });
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let (Some(_), Some(shape)) = (self.frame, cx.frame.shape) else {
            return;
        };
        let ids = self.ids;
        let (Some(pass), Some(canvas)) = (ids.pass, ids.canvas) else {
            return;
        };
        let _ = graph
            .add(cx.table, pass)
            .read(ids.ids)
            .read(ids.count)
            .read(ids.terrain)
            .read(ids.depth)
            .color(canvas, Access::Clear(Clear::Color(shape.canvas_clear)));
    }

    fn encode(
        &mut self,
        _pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        let Some(view) = self.view.get() else {
            return 0;
        };
        let (_, group) = view.group.as_ref().expect("bound_in_finish");
        let mut pass = cx.render_pass(encoder, "engine/debug-view", None);
        pass.set_pipeline(&view.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..3, 0..1);
        0
    }

    fn poll(&mut self) -> bool {
        self.view.poll()
    }

    fn disable(&mut self) -> bool {
        self.view.disable()
    }

    fn disabled(&self) -> bool {
        self.view.disabled()
    }

    #[cfg(feature = "dev")]
    fn inject(&mut self) -> bool {
        self.view.inject()
    }

    #[cfg(feature = "dev")]
    fn encode_fault(&self, encoder: &mut wgpu::CommandEncoder) {
        self.view.encode_fault(encoder);
    }
}
