use engine_core::InstanceInput;
use engine_places::meshes::DEBUG_LEVEL_AT;
use engine_render::wgpu;

pub(crate) const RECORD_ATTRIBUTES: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
    format: wgpu::VertexFormat::Uint32x4,
    offset: 0,
    shader_location: InstanceInput::Draw.location(),
}];
pub(crate) const RECORD_STREAM: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: 0,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &RECORD_ATTRIBUTES,
};

const LEVELS: u32 = 16;
const RECORD_BYTES: u64 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DrawRecord {
    pub level: u32,
}

impl DrawRecord {
    pub(crate) fn of(level: usize) -> Self {
        Self {
            level: level.min(LEVELS as usize - 1) as u32,
        }
    }

    fn word(self) -> u32 {
        self.level.min(LEVELS - 1) << DEBUG_LEVEL_AT
    }

    pub(crate) fn offset(self) -> u64 {
        u64::from(self.level.min(LEVELS - 1)) * RECORD_BYTES
    }
}

pub(crate) fn record_words() -> Vec<u32> {
    let mut words = Vec::new();
    for level in 0..LEVELS {
        words.extend([DrawRecord { level }.word(), 0, 0, 0]);
    }
    words
}

pub(crate) fn records_bytes() -> u64 {
    u64::from(LEVELS) * RECORD_BYTES
}
