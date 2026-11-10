import package::engine::math::maximum_abs;
import package::engine::shading::ViewPixel;

@group(0) @binding(0) var<uniform> camera: CameraUniform;

override PASS_CLAMP_NEAR: bool = false;

fn view_clip(world: vec3<f32>) -> vec4<f32> {
    let clip = camera.view_projection * vec4<f32>(world, 1.0);
    if PASS_CLAMP_NEAR {
        return vec4<f32>(clip.xy, max(clip.z, 0.0), clip.w);
    }
    return clip;
}

fn eye_world(relative: vec3<f32>) -> vec3<f32> {
    return relative + camera.position.xyz;
}

fn view_pixel(position: vec4<f32>, previous: vec4<f32>) -> ViewPixel {
    let current = vec2<f32>(position.x * camera.jitter.z - 1.0, 1.0 - position.y * camera.jitter.w) - camera.jitter.xy;
    return ViewPixel(vec4<f32>(current, 0.0, 1.0), previous);
}

fn detail_distance(world: vec3<f32>) -> f32 {
    return length(world - camera.detail.xyz);
}

fn detail_across(world: vec3<f32>) -> f32 {
    return detail_distance(world) * camera.pixel_angle;
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

fn seen_share(pixels: f32) -> f32 {
    return saturate(1.0 + log2(max(pixels, 1.0e-6)));
}

fn material_bias() -> f32 {
    return camera.mip_bias;
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

fn noise_phase() -> f32 {
    return camera.noise_phase;
}

fn transparency_weighting() -> vec4<f32> {
    return camera.transparency;
}

fn fade_noise(position: vec2<f32>) -> f32 {
    return fract(fract(52.9829189 * fract(dot(position, vec2<f32>(0.06711056, 0.00583715))))
        + noise_phase());
}

fn pixel_noise(position: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(position, vec2<f32>(0.06711056, 0.00583715))));
}
