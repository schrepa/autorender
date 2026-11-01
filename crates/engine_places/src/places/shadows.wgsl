const DEBUG_SUN_NONE: u32 = 0u;
const DEBUG_SUN_LEVEL: u32 = 1u;
const DEBUG_SUN_HORIZON: u32 = 21u;
const DEBUG_SUN_BEYOND: u32 = 22u;
const DEBUG_SHADOWED: u32 = 0x80u;

struct KeptTexels {
    first: u32,
    count: u32,
    texels: vec4<f32>,
}

struct KeptLift {
    first: u32,
    lifts: vec4<f32>,
}

fn sun_shadowed() -> bool {
    return false;
}

fn sun_shadow(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>, crown: f32)
        -> f32 {
    return 1.0;
}

fn air_shadow(world: vec3<f32>) -> f32 {
    return 1.0;
}

fn shadow_lookup(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>)
        -> vec2<f32> {
    return vec2<f32>(1.0, -1.0);
}

fn kept_texels(world: vec3<f32>) -> KeptTexels {
    var kept: KeptTexels;
    return kept;
}

fn keep_lift(lift: KeptLift) {
}

fn kept_level(texel: f32, spacing: f32, levels: u32) -> u32 {
    return 0u;
}

fn sky_visibility(world: vec3<f32>, normal: vec3<f32>) -> f32 {
    return 1.0;
}

fn debug_level_offset() -> u32 {
    return 0u;
}
