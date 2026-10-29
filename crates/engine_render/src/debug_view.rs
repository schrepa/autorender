#![expect(
    clippy::expect_used,
    reason = "the pass binds the targets its frame made"
)]

use engine_core::{DebugView, DebugViewMode, RenderSnapshot};

use crate::bindings::DEBUG_VIEW;
use crate::gpu_layouts::constants::{
    DEBUG_CASTS, DEBUG_KIND_CARD, DEBUG_KIND_MESH, DEBUG_KIND_PLACED, DEBUG_KIND_PLACED_CARD,
    DEBUG_LEVEL_SHIFT, DEBUG_VIEW_CARDS, DEBUG_VIEW_CASTERS, DEBUG_VIEW_DETAIL, DEBUG_VIEW_KINDS,
    DEBUG_VIEW_LEVELS, DEBUG_VIEW_MATERIALS, DEBUG_VIEW_NORMALS, DEBUG_VIEW_OVERDRAW,
    DEBUG_VIEW_SHADOWS, DEBUG_VIEW_SKY, DEBUG_VIEW_TILES, DEBUG_VIEW_WATER_DEPTH,
    DEBUG_VIEW_WATER_EDDIES, DEBUG_VIEW_WATER_FROUDE, DEBUG_VIEW_WATER_REACH,
    DEBUG_VIEW_WATER_ROUGHNESS, DEBUG_VIEW_WATER_SPAN, DEBUG_VIEW_WATER_SPEED,
    DEBUG_VIEW_WATER_TURBULENCE, DEBUG_VIEW_WATER_WHITE,
};
use crate::gpu_layouts::debug_view::DebugViewUniform;
use crate::graph::{
    Access, Clear, FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl, ResourceId,
    Stage,
};
use crate::lazy::Lazy;
use crate::parts::{Create, Encode, GpuPart, PartDecl, PartEntry, Step, View};
use crate::shaders::entry::debug_view;
use crate::targets::{FrameSize, ScreenTarget};

pub(crate) const IDS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;
pub(crate) const COUNT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
pub(crate) const TERRAIN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;

pub(crate) fn color_targets() -> Vec<Option<wgpu::ColorTargetState>> {
    let add = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    vec![
        Some(IDS_FORMAT.into()),
        Some(wgpu::ColorTargetState {
            format: COUNT_FORMAT,
            blend: Some(wgpu::BlendState {
                color: add,
                alpha: add,
            }),
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(TERRAIN_FORMAT.into()),
    ]
}

pub(crate) const RECORD_SLOT: u32 = 4;

const RECORD_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![14 => Uint32x4];

pub(crate) fn record_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 0,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &RECORD_ATTRIBUTES,
    }
}

pub(crate) fn with_record<'a>(
    buffers: &[Option<wgpu::VertexBufferLayout<'a>>],
) -> Vec<Option<wgpu::VertexBufferLayout<'a>>> {
    let mut streams = buffers.to_vec();
    streams.resize(RECORD_SLOT as usize, None);
    streams.push(Some(record_layout()));
    streams
}

pub(crate) fn bind_record(
    pass: &mut wgpu::RenderPass<'_>,
    records: Option<&DebugViewPass>,
    record: DrawRecord,
) {
    if let Some(slice) = records.and_then(|records| records.record(record)) {
        pass.set_vertex_buffer(RECORD_SLOT, slice);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DrawKind {
    Mesh,
    Placed,
    Card,
    PlacedCard,
}

impl DrawKind {
    const ALL: [Self; 4] = [Self::Mesh, Self::Placed, Self::Card, Self::PlacedCard];

    fn kind(self) -> u32 {
        match self {
            Self::Mesh => DEBUG_KIND_MESH,
            Self::Placed => DEBUG_KIND_PLACED,
            Self::Card => DEBUG_KIND_CARD,
            Self::PlacedCard => DEBUG_KIND_PLACED_CARD,
        }
    }
}

const LEVELS: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DrawRecord {
    pub kind: DrawKind,
    pub level: u32,
    pub casts: bool,
}

impl DrawRecord {
    pub(crate) fn of(kind: DrawKind, level: usize, casts: bool) -> Self {
        Self {
            kind,
            level: level.min(LEVELS as usize - 1) as u32,
            casts,
        }
    }

    fn index(self) -> u64 {
        let kind = DrawKind::ALL
            .iter()
            .position(|kind| *kind == self.kind)
            .unwrap_or_default() as u64;
        (kind * u64::from(LEVELS) + u64::from(self.level.min(LEVELS - 1))) * 2
            + u64::from(self.casts)
    }

    fn word(self) -> u32 {
        let mut word = self.kind.kind() | (self.level.min(LEVELS - 1) << DEBUG_LEVEL_SHIFT);
        if self.casts {
            word |= DEBUG_CASTS;
        }
        word
    }
}

const RECORD_BYTES: u64 = 16;

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

pub(crate) fn shows_water(view: DebugViewMode) -> bool {
    matches!(
        view,
        DebugViewMode::WaterDepth
            | DebugViewMode::WaterSpeed
            | DebugViewMode::WaterFroude
            | DebugViewMode::WaterTurbulence
            | DebugViewMode::WaterWhite
            | DebugViewMode::WaterReach
            | DebugViewMode::WaterSpan
            | DebugViewMode::WaterEddies
            | DebugViewMode::WaterRoughness
    )
}

pub(crate) fn water_constants(water_views: bool) -> [(&'static str, f64); 1] {
    [("DEBUG_WATER_VIEWS", if water_views { 1.0 } else { 0.0 })]
}

struct Targets {
    textures: [wgpu::Texture; 3],
    ids: wgpu::TextureView,
    count: wgpu::TextureView,
    terrain: wgpu::TextureView,
    group: wgpu::BindGroup,
}

impl Drop for Targets {
    fn drop(&mut self) {
        for texture in &self.textures {
            texture.destroy();
        }
    }
}

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "debug_view",
        brick: "debug_view",
        passes: &[PassDecl {
            name: "debug_view",
            stage: Stage::Composite,
        }],
        resources: &[
            ResourceDecl {
                name: "debug_ids",
                lifetime: Lifetime::Frame,
            },
            ResourceDecl {
                name: "debug_count",
                lifetime: Lifetime::Frame,
            },
            ResourceDecl {
                name: "debug_terrain",
                lifetime: Lifetime::Frame,
            },
        ],
        hooks: &[],
        contributes: &[],
        effects: &[],
        labels: &["engine/debug-view"],
        lazy: true,
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
pub(crate) struct DebugViewPass {
    view: Lazy<DebugTargets>,
    frame: Option<DebugViewFrame>,
    ids: Ids,
}

impl DebugViewPass {
    pub(crate) fn shows(&self, snapshot: &RenderSnapshot, forced: Option<bool>) -> bool {
        let showing = forced.unwrap_or(snapshot.environment.debug_view.showing());
        showing && !self.view.disabled()
    }

    pub(crate) fn records(&self) -> Option<&Self> {
        self.view.get().map(|_| self)
    }

    pub(crate) fn record(&self, record: DrawRecord) -> Option<wgpu::BufferSlice<'_>> {
        Some(self.view.get()?.record(record))
    }

    #[cfg(feature = "dev")]
    pub(crate) fn textures(&self) -> Option<[&wgpu::Texture; 3]> {
        self.view.get()?.textures()
    }
}

struct DebugTargets {
    layout: crate::pipelines::GroupLayout,
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    records: wgpu::Buffer,
    targets: ScreenTarget<Targets>,
}

#[derive(Clone, Copy)]
struct DebugViewFrame;

impl DebugTargets {
    fn new(device: &wgpu::Device, canvas: wgpu::TextureFormat) -> Self {
        let layout = crate::pipelines::layout(device, &DEBUG_VIEW);
        let pipeline_layout =
            crate::pipelines::layout_of(device, &crate::pipeline_layouts::DEBUG_VIEW_PIPELINES);
        let pipeline = crate::pipelines::fullscreen(
            device,
            &crate::pipelines::Fullscreen {
                label: "engine/debug-view",
                layout: &pipeline_layout,
                vertex: debug_view::vs_fullscreen,
                fragment: debug_view::fs_debug_view,
                constants: &[],
                targets: &[Some(canvas.into())],
                depth: None,
            },
        );
        let mut words = Vec::new();
        for kind in DrawKind::ALL {
            for level in 0..LEVELS {
                for casts in [false, true] {
                    let record = DrawRecord { kind, level, casts };
                    words.extend([record.word(), 0, 0, 0]);
                }
            }
        }
        let records = crate::memory::buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("engine/debug-view-records"),
                contents: bytemuck::cast_slice(&words),
                usage: wgpu::BufferUsages::VERTEX,
            },
        );
        Self {
            layout,
            pipeline,
            uniform: crate::buffers::uniform(
                device,
                "engine/debug-view-uniform",
                size_of::<DebugViewUniform>() as u64,
            ),
            records,
            targets: ScreenTarget::default(),
        }
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        size: FrameSize,
        settings: DebugView,
        sun: [f32; 3],
    ) -> DebugViewFrame {
        self.begin(device, size);
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

    fn begin(&mut self, device: &wgpu::Device, size: FrameSize) {
        if !self.targets.resizes(size) {
            return;
        }
        let texture = |label, format| {
            crate::targets::texture_2d(
                device,
                label,
                size.pixels(),
                1,
                1,
                format,
                crate::targets::RENDER_TARGET | wgpu::TextureUsages::COPY_SRC,
            )
        };
        let textures = [
            texture("engine/debug-view-ids", IDS_FORMAT),
            texture("engine/debug-view-count", COUNT_FORMAT),
            texture("engine/debug-view-terrain", TERRAIN_FORMAT),
        ];
        let [ids, count, terrain] = textures
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
        let group = crate::pipelines::group(
            device,
            "engine/debug-view-group",
            &DEBUG_VIEW,
            &self.layout,
            &[
                (
                    "debug_view",
                    wgpu::BindingResource::Buffer(self.uniform.as_entire_buffer_binding()),
                ),
                ("debug_ids", crate::pipelines::view(&ids)),
                ("debug_count", crate::pipelines::view(&count)),
                ("debug_terrain", crate::pipelines::view(&terrain)),
            ],
        );
        self.targets.set(
            size,
            Targets {
                textures,
                ids,
                count,
                terrain,
                group,
            },
        );
    }

    #[cfg(feature = "dev")]
    fn textures(&self) -> Option<[&wgpu::Texture; 3]> {
        let targets = self.targets.get()?;
        Some(targets.textures.each_ref())
    }

    fn views(&self) -> Option<[&wgpu::TextureView; 3]> {
        let targets = self.targets.get()?;
        Some([&targets.ids, &targets.count, &targets.terrain])
    }

    fn record(&self, record: DrawRecord) -> wgpu::BufferSlice<'_> {
        let offset = record.index() * RECORD_BYTES;
        self.records.slice(offset..offset + RECORD_BYTES)
    }

    fn capacity_bytes(&self) -> u64 {
        self.targets.size().map_or(0, |size| {
            u64::from(size.width) * u64::from(size.height) * 26 + self.records.size()
        })
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

    fn finish(&mut self, cx: &mut Step<'_>) {
        if !cx.frame.kind.debug {
            self.frame = None;
            return;
        }
        let snapshot = cx.frame.snapshot;
        let core = &*cx.core;
        let (device, queue) = (&core.device, &core.queue);
        let (size, format) = (core.display.size(), core.canvas_format());
        let sun = snapshot
            .environment
            .atmosphere
            .sun_direction
            .normalize_or_zero()
            .to_array();
        self.view.create(device, "engine/debug-view", || {
            DebugTargets::new(device, format)
        });
        self.frame = self
            .view
            .get_mut()
            .map(|view| view.prepare(device, queue, size, snapshot.environment.debug_view, sun));
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
        let targets = view.targets.get().expect("sized_in_prepare");
        let mut pass = cx.render_pass(encoder, "engine/debug-view", None);
        pass.set_pipeline(&view.pipeline);
        pass.set_bind_group(0, &targets.group, &[]);
        pass.draw(0..3, 0..1);
        0
    }

    fn attachment(&self, resource: &str) -> Option<wgpu::TextureView> {
        let [ids, count, terrain] = self.view.get()?.views()?;
        match resource {
            "debug_ids" => Some(ids.clone()),
            "debug_count" => Some(count.clone()),
            "debug_terrain" => Some(terrain.clone()),
            _ => None,
        }
    }

    fn capacity_bytes(&self) -> u64 {
        self.view.get().map_or(0, DebugTargets::capacity_bytes)
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
