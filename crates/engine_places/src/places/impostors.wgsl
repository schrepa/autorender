struct ImpostorRecord {
    sphere: vec4<f32>,
    extent: vec3<f32>,
    layer: u32,
    emissive: vec3<f32>,
    flags: u32,
}

struct ImpostorBasis {
    right: vec3<f32>,
    up: vec3<f32>,
}

fn impostor_frames(flags: u32) -> u32 {
    return 2u;
}

fn impostor_flags_texels(flags: u32) -> u32 {
    return 0u;
}

fn impostor_flags_hemisphere(flags: u32) -> bool {
    return false;
}

fn impostor_flags_thin(flags: u32) -> bool {
    return false;
}

fn impostor_grid(d: vec3<f32>, hemisphere: bool) -> vec2<f32> {
    return vec2<f32>(0.0);
}

fn impostor_frame_direction(frame: u32, frames: u32, hemisphere: bool) -> vec3<f32> {
    return vec3<f32>(0.0, 1.0, 0.0);
}

fn impostor_basis(d: vec3<f32>) -> ImpostorBasis {
    var basis: ImpostorBasis;
    return basis;
}

fn impostor_facing(slope: vec2<f32>) -> f32 {
    return 1.0;
}

fn impostor_met(sampled: f32, covered: f32, crossing: f32, facing: f32) -> f32 {
    return 0.0;
}

fn impostor_albedo(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn impostor_emissive(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn impostor_geometry(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn impostor_surface(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn impostor_size() -> u32 {
    return 1u;
}

fn impostor_levels() -> u32 {
    return 1u;
}
