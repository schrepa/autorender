use engine_core::GpuBits;
use engine_render::debug::DebugRecord;
use engine_render::wgpu;

pub(crate) const RECORD_ATTRIBUTES: [wgpu::VertexAttribute; 1] =
    wgpu::vertex_attr_array![14 => Uint32x4];
pub(crate) const RECORD_STREAM: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: 0,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &RECORD_ATTRIBUTES,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DrawKind {
    Mesh,
    Card,
}

impl DrawKind {
    const ALL: [Self; 2] = [Self::Mesh, Self::Card];
}

const LEVELS: u32 = 16;
const RECORD_BYTES: u64 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DrawRecord {
    pub kind: DrawKind,
    pub level: u32,
}

impl DrawRecord {
    pub(crate) fn of(kind: DrawKind, level: usize) -> Self {
        Self {
            kind,
            level: level.min(LEVELS as usize - 1) as u32,
        }
    }

    fn index(self) -> u64 {
        let kind = DrawKind::ALL
            .iter()
            .position(|kind| *kind == self.kind)
            .unwrap_or_default() as u64;
        kind * u64::from(LEVELS) + u64::from(self.level.min(LEVELS - 1))
    }

    fn word(self) -> u32 {
        DebugRecord {
            kind: 0,
            level: self.level.min(LEVELS - 1),
            fading: false,
            card: self.kind == DrawKind::Card,
            source: 0,
        }
        .pack()
    }

    pub(crate) fn offset(self) -> u64 {
        self.index() * RECORD_BYTES
    }
}

pub(crate) fn record_words() -> Vec<u32> {
    let mut words = Vec::new();
    for kind in DrawKind::ALL {
        for level in 0..LEVELS {
            let record = DrawRecord { kind, level };
            words.extend([record.word(), 0, 0, 0]);
        }
    }
    words
}

pub(crate) fn records_bytes() -> u64 {
    DrawKind::ALL.len() as u64 * u64::from(LEVELS) * RECORD_BYTES
}
