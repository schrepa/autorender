import package::engine::math::maximum_abs;

struct Camera {
    view_projection: mat4x4<f32>,
    previous_view_projection: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
    position: vec4<f32>,
    jitter: vec4<f32>,
    options: vec4<f32>,
    wind: vec4<f32>,
    region: vec4<f32>,
    detail: vec4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;

const PI: f32 = 3.14159265;

fn eye_world(relative: vec3<f32>) -> vec3<f32> {
    return relative + camera.position.xyz;
}

const DETAIL_EDGE_ON: f32 = 0.0625;

const DETAIL_ANISOTROPY: f32 = 16.0;

fn detail_distance(world: vec3<f32>) -> f32 {
    return length(world - camera.detail.xyz);
}

fn detail_across(world: vec3<f32>) -> f32 {
    return detail_distance(world) * camera.options.y;
}

fn detail_span(world: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
    let across = detail_across(world);
    let toward = world - camera.detail.xyz;
    let ray = toward / max(length(toward), 1.0e-6);
    let cosine = dot(normal, ray);
    let facing = max(abs(cosine), DETAIL_EDGE_ON);
    let laid = ray - normal * cosine;
    let way = laid.xz * inverseSqrt(max(dot(laid, laid), 1.0e-12));
    return vec4<f32>(across, across / facing, way);
}

fn detail_resolved(span: vec2<f32>) -> f32 {
    return max(min(span.x, span.y), max(span.x, span.y) / DETAIL_ANISOTROPY);
}

struct DetailLevel {
    level: u32,
    blend: f32,
}

fn detail_level(span: f32, base: f32, levels: u32) -> DetailLevel {
    let coarsest = f32(max(levels, 1u) - 1u);
    let at = clamp(log2(max(span / max(base, 1.0e-30), 1.0e-30)), 0.0, coarsest);
    var chosen: DetailLevel;
    chosen.level = u32(floor(at));
    chosen.blend = saturate(2.0 * (at - floor(at)) - 1.0);
    return chosen;
}

fn material_bias() -> f32 {
    return camera.options.w;
}

fn material_detail() -> f32 {
    return camera.region.z;
}

fn stable_normalize(value: vec3<f32>) -> vec3<f32> {
    let scale = maximum_abs(value);
    if scale == 0.0 { // optics: a zero vector's guard, whatever it measures
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(value / scale);
}

fn fade_noise(position: vec2<f32>) -> f32 {
    return fract(fract(52.9829189 * fract(dot(position, vec2<f32>(0.06711056, 0.00583715))))
        + camera.options.z);
}

fn pixel_noise(position: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(position, vec2<f32>(0.06711056, 0.00583715))));
}
