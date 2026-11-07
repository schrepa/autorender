use engine_render::{AttachPoint, TargetDecl, TargetName, Written, wgpu};

pub(crate) const JITTER_SAMPLES: u64 = 8;

pub(crate) const MOTION: TargetDecl = TargetDecl {
    name: TargetName::MOTION,
    format: wgpu::TextureFormat::Rg16Float,
    written: Written::At(&[AttachPoint::Depth, AttachPoint::Opaque, AttachPoint::Sky]),
    module: Some(crate::shaders::taa::motion::MODULE),
};

fn halton(mut index: u32, base: u32) -> f32 {
    let (mut scale, mut value) = (1.0, 0.0);
    while index > 0 {
        scale /= base as f32;
        value += scale * (index % base) as f32;
        index /= base;
    }
    value
}

pub(crate) fn jitter_offset(frame: u64) -> [f32; 2] {
    let index = (frame % JITTER_SAMPLES) as u32 + 1;
    [halton(index, 2) - 0.5, halton(index, 3) - 0.5]
}

pub(crate) fn turn_of(frame: u64) -> f32 {
    let turn = (frame % 1024) as f32 * 0.618_034;
    turn - turn.floor()
}
