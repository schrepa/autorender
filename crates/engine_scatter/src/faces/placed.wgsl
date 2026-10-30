struct Record {
    center: vec3<f32>,
    radius: f32,
    group: u32,
    scale: f32,
    terrain: u32,
    sink: f32,
    state: u32,
    pad: array<u32, 3>,
}

struct Pose {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

const DEAD: u32 = 0xffffffffu;
const HIDDEN: u32 = 0x80000000u;

fn placed_shown(record: Record) -> bool {
    return record.group != DEAD && (record.terrain & HIDDEN) == 0u;
}
