struct SunAt {
    world: vec3<f32>,
    normal: vec3<f32>,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    softness: f32,
    ground: bool,
    lift_first: u32,
    lifts: vec4<f32>,
}

struct SunOpen {
    blocked: f32,
    passing: f32,
}

fn sun_open(open: SunOpen, at: SunAt) -> SunOpen {
    return open;
}

fn sun_flags(flags: u32, at: SunAt) -> u32 {
    return flags;
}
