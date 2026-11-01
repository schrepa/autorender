use engine_render::debug::{DEBUG_CASTS, DEBUG_KIND_CARD, DEBUG_KIND_MESH, DEBUG_LEVEL_SHIFT};
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

    fn kind(self) -> u32 {
        match self {
            Self::Mesh => DEBUG_KIND_MESH,
            Self::Card => DEBUG_KIND_CARD,
        }
    }
}

const LEVELS: u32 = 16;
const RECORD_BYTES: u64 = 16;

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

    pub(crate) fn offset(self) -> u64 {
        self.index() * RECORD_BYTES
    }
}

pub(crate) fn record_words() -> Vec<u32> {
    let mut words = Vec::new();
    for kind in DrawKind::ALL {
        for level in 0..LEVELS {
            for casts in [false, true] {
                let record = DrawRecord { kind, level, casts };
                words.extend([record.word(), 0, 0, 0]);
            }
        }
    }
    words
}

pub(crate) fn records_bytes() -> u64 {
    DrawKind::ALL.len() as u64 * u64::from(LEVELS) * 2 * RECORD_BYTES
}
