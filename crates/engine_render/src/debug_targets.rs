use crate::cached::Versioned;
use crate::gpu_layouts::constants::{
    DEBUG_CASTS, DEBUG_KIND_CARD, DEBUG_KIND_MESH, DEBUG_KIND_PLACED, DEBUG_KIND_PLACED_CARD,
    DEBUG_LEVEL_SHIFT,
};
use crate::graph::Targets;
use crate::parts::View;
use crate::renderer::Core;
use crate::targets::{FrameSize, ScreenTarget};

pub const IDS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;
pub const COUNT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
pub const TERRAIN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;

pub fn color_targets() -> Vec<Option<wgpu::ColorTargetState>> {
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

pub const RECORD_SLOT: u32 = 4;

const RECORD_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![14 => Uint32x4];

pub fn record_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 0,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &RECORD_ATTRIBUTES,
    }
}

pub fn with_record<'a>(
    buffers: &[Option<wgpu::VertexBufferLayout<'a>>],
) -> Vec<Option<wgpu::VertexBufferLayout<'a>>> {
    let mut streams = buffers.to_vec();
    streams.resize(RECORD_SLOT as usize, None);
    streams.push(Some(record_layout()));
    streams
}

pub fn bind_record(
    pass: &mut wgpu::RenderPass<'_>,
    records: Option<&DrawRecords>,
    record: DrawRecord,
) {
    if let Some(records) = records {
        pass.set_vertex_buffer(RECORD_SLOT, records.record(record));
    }
}

pub fn debug_records<'a>(cx: &View<'a>) -> Option<&'a DrawRecords> {
    cx.frame
        .shape
        .is_some_and(|shape| shape.targets == Targets::Debug)
        .then_some(&cx.core.debug.records)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawKind {
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
pub struct DrawRecord {
    pub kind: DrawKind,
    pub level: u32,
    pub casts: bool,
}

impl DrawRecord {
    pub fn of(kind: DrawKind, level: usize, casts: bool) -> Self {
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

pub struct DrawRecords {
    records: wgpu::Buffer,
}

impl DrawRecords {
    fn new(device: &wgpu::Device) -> Self {
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
        Self { records }
    }

    pub fn record(&self, record: DrawRecord) -> wgpu::BufferSlice<'_> {
        let offset = record.index() * RECORD_BYTES;
        self.records.slice(offset..offset + RECORD_BYTES)
    }
}

pub struct DebugTargets {
    textures: [wgpu::Texture; 3],
    pub ids: wgpu::TextureView,
    pub count: wgpu::TextureView,
    pub terrain: wgpu::TextureView,
}

impl Drop for DebugTargets {
    fn drop(&mut self) {
        for texture in &self.textures {
            texture.destroy();
        }
    }
}

impl DebugTargets {
    pub fn textures(&self) -> [&wgpu::Texture; 3] {
        self.textures.each_ref()
    }

    fn view(&self, resource: &str) -> Option<&wgpu::TextureView> {
        match resource {
            "debug_ids" => Some(&self.ids),
            "debug_count" => Some(&self.count),
            "debug_terrain" => Some(&self.terrain),
            _ => None,
        }
    }
}

pub struct DebugFrame {
    pub(crate) records: DrawRecords,
    targets: ScreenTarget<DebugTargets>,
}

impl DebugFrame {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            records: DrawRecords::new(device),
            targets: ScreenTarget::default(),
        }
    }

    pub(crate) fn prepare(&mut self, device: &wgpu::Device, size: FrameSize) {
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
        self.targets.set(
            size,
            DebugTargets {
                textures,
                ids,
                count,
                terrain,
            },
        );
    }

    pub(crate) fn release(&mut self) {
        self.targets.release();
    }

    pub(crate) fn attachment(&self, resource: &str) -> Option<wgpu::TextureView> {
        self.targets.get()?.view(resource).cloned()
    }

    #[cfg(feature = "dev")]
    pub(crate) fn textures(&self) -> Option<[&wgpu::Texture; 3]> {
        Some(self.targets.get()?.textures())
    }
}

impl Core {
    pub fn draw_record(&self, record: DrawRecord) -> wgpu::BufferSlice<'_> {
        self.debug.records.record(record)
    }

    pub fn debug_targets(&self) -> Option<&Versioned<DebugTargets>> {
        self.debug.targets.get()
    }
}
