import package::generated::effects::EffectsFrame;

struct EffectItem {
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
    p4: vec4<f32>,
    p5: vec4<f32>,
    p6: vec4<f32>,
    p7: vec4<f32>,
}

@group(2) @binding(0) var<storage, read> effect_items: array<EffectItem>;
@group(2) @binding(1) var<uniform> effects_frame: EffectsFrame;
@group(2) @binding(2) var effects_noise_map: texture_3d<f32>;
@group(2) @binding(3) var effects_noise_sampler: sampler;

const EFFECT_HIDDEN: vec4<f32> = vec4<f32>(2.0, 2.0, 2.0, 1.0);
const EFFECT_TAU: f32 = 6.2831853;
const EFFECT_INV_24: f32 = 1.0 / 16777216.0;
const EFFECT_NOISE_INV: f32 = 1.0 / 32.0;

fn effect_hash(value: u32) -> u32 {
    var x = value * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}

fn effect_noise_raw(seed: u32) -> f32 {
    return f32(effect_hash(seed) >> 8u);
}

fn effect_random(seed: u32) -> f32 {
    return effect_noise_raw(seed) * EFFECT_INV_24;
}

fn effect_direction(seed: u32) -> vec3<f32> {
    let z = effect_random(seed) * 2.0 - 1.0;
    let angle = effect_random(seed + 1u) * EFFECT_TAU;
    let ring = sqrt(max(1.0 - z * z, 0.0));
    return vec3<f32>(ring * cos(angle), z, ring * sin(angle));
}

fn effect_corner(vertex: u32) -> vec2<f32> {
    let corner = vertex % 6u;
    return vec2<f32>(f32((50u >> corner) & 1u), f32((44u >> corner) & 1u)) * 2.0 - 1.0;
}

fn effect_axis(index: u32) -> vec3<f32> {
    return vec3<f32>(select(0.0, 1.0, index == 0u), select(0.0, 1.0, index == 1u),
        select(0.0, 1.0, index == 2u));
}

struct EffectBoxCorner {
    local: vec3<f32>,
    normal: vec3<f32>,
}

fn effect_box_corner(vertex: u32) -> EffectBoxCorner {
    let face = vertex / 6u;
    let axis = face / 2u;
    let sign = 1.0 - 2.0 * f32(face % 2u);
    let first = effect_axis((axis + 1u) % 3u);
    let second = effect_axis((axis + 2u) % 3u);
    let across = select(second, first, sign > 0.0);
    let up = select(first, second, sign > 0.0);
    let unit = effect_corner(vertex);
    let normal = effect_axis(axis) * sign;
    return EffectBoxCorner(normal + across * unit.x + up * unit.y, normal);
}

fn effect_lattice(cell: vec3<f32>) -> f32 {
    let bits = vec3<u32>(vec3<i32>(cell) + vec3<i32>(4096));
    return effect_random((bits.x * 73856093u) ^ (bits.y * 19349663u) ^ (bits.z * 83492791u));
}

fn effect_noise_2d(point: vec2<f32>, z_int: f32) -> f32 {
    let cell = floor(point);
    let part = point - cell;
    let blend = part * part * (3.0 - 2.0 * part);
    return textureSampleLevel(
        effects_noise_map,
        effects_noise_sampler,
        vec3<f32>(cell + blend + 0.5, z_int + 0.5) * EFFECT_NOISE_INV,
        0.0,
    ).r;
}

fn effect_noise(point: vec3<f32>) -> f32 {
    let cell = floor(point);
    let part = point - cell;
    let blend = part * part * (3.0 - 2.0 * part);
    return textureSampleLevel(
        effects_noise_map,
        effects_noise_sampler,
        (cell + blend + 0.5) * EFFECT_NOISE_INV,
        0.0,
    ).r;
}

fn effect_axes(item: EffectItem) -> mat3x3<f32> {
    let x = item.p4.xyz;
    let y = item.p5.xyz;
    return mat3x3<f32>(x, y, cross(x, y));
}

fn effect_seconds() -> f32 {
    return effects_frame.clock.x;
}

fn effect_lit(color: vec3<f32>, normal: vec3<f32>, field: vec4<f32>) -> vec3<f32> {
    let sun = effects_frame.sun;
    if sun.w == 0.0 {
        return color;
    }
    let toward = saturate(dot(normal, sun.xyz) * 0.5 + 0.5);
    let sky = mix(0.6, 1.0, saturate(normal.y * 0.5 + 0.5));
    return color * (field.rgb * sky + effects_frame.sun_color.rgb * toward * 0.5 * field.a);
}
