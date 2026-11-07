import package::generated::impostors::{impostor_flags_frames as flags_frames,
    impostor_flags_hemisphere as flags_hemisphere, impostor_flags_texels as flags_texels,
    impostor_flags_thin as flags_thin};
import package::place::impostors::ImpostorBasis;

@group(0) @binding(0) var atlas_albedo: texture_2d_array<f32>;
@group(0) @binding(0) var atlas_emissive: texture_2d_array<f32>;
@group(0) @binding(0) var atlas_geometry: texture_2d_array<f32>;
@group(0) @binding(0) var atlas_surface: texture_2d_array<f32>;
@group(0) @binding(0) var atlas_sampler: sampler;

fn impostor_frames(flags: u32) -> u32 {
    return flags_frames(flags);
}

fn impostor_flags_texels(flags: u32) -> u32 {
    return flags_texels(flags);
}

fn impostor_flags_hemisphere(flags: u32) -> bool {
    return flags_hemisphere(flags);
}

fn impostor_flags_thin(flags: u32) -> bool {
    return flags_thin(flags);
}

fn octahedral(d: vec3<f32>) -> vec2<f32> {
    let n = d / (abs(d.x) + abs(d.y) + abs(d.z));
    let sign = select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xz >= vec2<f32>(0.0));
    return select(n.xz, (vec2<f32>(1.0) - abs(n.zx)) * sign, n.y < 0.0);
}

fn octahedral_direction(p: vec2<f32>) -> vec3<f32> {
    var d = vec3<f32>(p.x, 1.0 - abs(p.x) - abs(p.y), p.y);
    let fold = max(-d.y, 0.0);
    d.x += select(fold, -fold, d.x >= 0.0);
    d.z += select(fold, -fold, d.z >= 0.0);
    return normalize(d);
}

fn hemi_octahedral(d: vec3<f32>) -> vec2<f32> {
    let above = vec3<f32>(d.x, max(d.y, 0.0), d.z);
    let n = above / max(abs(above.x) + above.y + abs(above.z), 1.0e-6);
    return vec2<f32>(n.x + n.z, n.x - n.z);
}

fn hemi_direction(p: vec2<f32>) -> vec3<f32> {
    let x = 0.5 * (p.x + p.y);
    let z = 0.5 * (p.x - p.y);
    return normalize(vec3<f32>(x, 1.0 - abs(x) - abs(z), z));
}

fn impostor_grid(d: vec3<f32>, hemisphere: bool) -> vec2<f32> {
    if hemisphere {
        return hemi_octahedral(d);
    }
    return octahedral(d);
}

fn impostor_frame_direction(frame: u32, frames: u32, hemisphere: bool) -> vec3<f32> {
    let cell = vec2<f32>(f32(frame % frames), f32(frame / frames));
    let p = cell / f32(frames - 1u) * 2.0 - 1.0;
    if hemisphere {
        return hemi_direction(p);
    }
    return octahedral_direction(p);
}

fn impostor_basis(d: vec3<f32>) -> ImpostorBasis {
    var hint = vec3<f32>(0.0, 1.0, 0.0);
    if abs(d.y) > 0.999 {
        hint = vec3<f32>(0.0, 0.0, -sign(d.y));
    }
    let right = normalize(cross(hint, d));
    return ImpostorBasis(right, cross(d, right));
}

fn impostor_facing(slope: vec2<f32>) -> f32 {
    return sqrt(max(1.0 - dot(slope, slope), 0.01));
}

fn impostor_met(sampled: f32, covered: f32, crossing: f32, facing: f32) -> f32 {
    return (2.0 * sampled - covered * (1.0 + crossing)) / facing;
}

fn impostor_albedo(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return textureSampleLevel(atlas_albedo, atlas_sampler, at, layer, level);
}

fn impostor_emissive(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return textureSampleLevel(atlas_emissive, atlas_sampler, at, layer, level);
}

fn impostor_geometry(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return textureSampleLevel(atlas_geometry, atlas_sampler, at, layer, level);
}

fn impostor_surface(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return textureSampleLevel(atlas_surface, atlas_sampler, at, layer, level);
}

fn impostor_size() -> u32 {
    return textureDimensions(atlas_albedo).x;
}

fn impostor_levels() -> u32 {
    return textureNumLevels(atlas_albedo);
}
