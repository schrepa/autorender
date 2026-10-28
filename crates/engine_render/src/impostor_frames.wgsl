const IMPOSTOR_HEMISPHERE: u32 = 0x10000u;
const IMPOSTOR_MASKED: u32 = 0x20000u;
const IMPOSTOR_FRAMES_SHIFT: u32 = 19u;

fn impostor_frames(flags: u32) -> u32 {
    return (flags >> IMPOSTOR_FRAMES_SHIFT) & 31u;
}

const IMPOSTOR_FADING: u32 = 0x80000000u;
const IMPOSTOR_FADING_OUT: u32 = 0x40000000u;
const IMPOSTOR_OVER: u32 = 0x8000u;

fn impostor_index(batch: u32) -> u32 {
    return select(batch, batch & 0x7fffu, (batch & IMPOSTOR_FADING) != 0u);
}

fn impostor_over(batch: u32) -> bool {
    return (batch & IMPOSTOR_FADING) != 0u && (batch & IMPOSTOR_OVER) != 0u;
}

fn impostor_fade(batch: u32) -> f32 {
    if (batch & IMPOSTOR_FADING) == 0u {
        return 1.0;
    }
    return f32((batch >> 16u) & 0x3fffu) / 16383.0;
}

struct ImpostorRecord {
    sphere: vec4<f32>,
    extent: vec3<f32>,
    layer: u32,
    emissive: vec3<f32>,
    flags: u32,
}

fn impostor_octahedral(d: vec3<f32>) -> vec2<f32> {
    let n = d / (abs(d.x) + abs(d.y) + abs(d.z));
    let sign = select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xz >= vec2<f32>(0.0));
    return select(n.xz, (vec2<f32>(1.0) - abs(n.zx)) * sign, n.y < 0.0);
}

fn impostor_direction(p: vec2<f32>) -> vec3<f32> {
    var d = vec3<f32>(p.x, 1.0 - abs(p.x) - abs(p.y), p.y);
    let fold = max(-d.y, 0.0);
    d.x += select(fold, -fold, d.x >= 0.0);
    d.z += select(fold, -fold, d.z >= 0.0);
    return normalize(d);
}

fn impostor_hemi_octahedral(d: vec3<f32>) -> vec2<f32> {
    let above = vec3<f32>(d.x, max(d.y, 0.0), d.z);
    let n = above / max(abs(above.x) + above.y + abs(above.z), 1.0e-6);
    return vec2<f32>(n.x + n.z, n.x - n.z);
}

fn impostor_hemi_direction(p: vec2<f32>) -> vec3<f32> {
    let x = 0.5 * (p.x + p.y);
    let z = 0.5 * (p.x - p.y);
    return normalize(vec3<f32>(x, 1.0 - abs(x) - abs(z), z));
}

fn impostor_grid(d: vec3<f32>, hemisphere: bool) -> vec2<f32> {
    if hemisphere {
        return impostor_hemi_octahedral(d);
    }
    return impostor_octahedral(d);
}

fn impostor_frame_direction(frame: u32, frames: u32, hemisphere: bool) -> vec3<f32> {
    let cell = vec2<f32>(f32(frame % frames), f32(frame / frames));
    let p = cell / f32(frames - 1u) * 2.0 - 1.0;
    if hemisphere {
        return impostor_hemi_direction(p);
    }
    return impostor_direction(p);
}

struct ImpostorBasis {
    right: vec3<f32>,
    up: vec3<f32>,
}

fn impostor_basis(d: vec3<f32>) -> ImpostorBasis {
    var hint = vec3<f32>(0.0, 1.0, 0.0);
    if abs(d.y) > 0.999 {
        hint = vec3<f32>(0.0, 0.0, -sign(d.y));
    }
    let right = normalize(cross(hint, d));
    return ImpostorBasis(right, cross(d, right));
}

const IMPOSTOR_CUTOFF: f32 = 0.5;

fn impostor_facing(slope: vec2<f32>) -> f32 {
    return sqrt(max(1.0 - dot(slope, slope), 0.01));
}

fn impostor_met(sampled: f32, covered: f32, crossing: f32, facing: f32) -> f32 {
    return (2.0 * sampled - covered * (1.0 + crossing)) / facing;
}
