const FOLD_COST: u32 = 0u;

struct SunAt {
    world: vec3<f32>,
    normal: vec3<f32>,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    softness: f32,
    kind: u32,
    lift_first: u32,
    lifts: vec4<f32>,
}

fn sun_open(open: f32, at: SunAt) -> f32 {
    return open;
}

fn sun_air(open: f32, world: vec3<f32>) -> f32 {
    return open;
}

fn sun_flags(flags: u32, at: SunAt) -> u32 {
    return flags;
}
