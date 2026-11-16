import package::engine::detail::{DetailLevel, ViewDetail, across, detail_level, fits, handover,
    pixels, seen_share};
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

// optics: FLOAT_GAP is the gap between neighbouring 32-bit floats as a share of their size, 2^-23.
const FLOAT_GAP: f32 = 1.1920929e-7;
// optics: DEPTH_ROUNDINGS counts what can shift a drawn depth against the depth worked out below:
// optics: the draw's product with the view-projection (3 products, 3 sums), the receiver's point
// optics: interpolated per coordinate (3 products, 2 sums, 1 quotient), 1 sum moving it back, and
// optics: the product below (3 products, 3 sums). Each is at most one gap.
const DEPTH_ROUNDINGS: f32 = 19.0;

fn eye_nearer(world: vec3<f32>, held: f32) -> bool {
    let toward = vec3<f32>(camera.view_projection[0].w, camera.view_projection[1].w,
        camera.view_projection[2].w);
    let depth = dot(toward, world - camera.position.xyz);
    let size = depth * camera.pixel_angle; // optics: one pixel's size at the receiver
    let gap = FLOAT_GAP * (maximum_abs(camera.position.xyz) + depth); // optics: the largest value on the way sets the gap
    let slack = size + DEPTH_ROUNDINGS * gap;
    let back = camera.view_projection * vec4<f32>(world + toward * slack, 1.0); // optics: the product that drew the held depth
    return quantizeToF16(back.z / back.w) > held; // optics: both sides round as the store does, which keeps their order
}

fn view_pixel(position: vec4<f32>, previous: vec4<f32>) -> ViewPixel {
    let current = vec2<f32>(position.x * camera.jitter.z - 1.0, 1.0 - position.y * camera.jitter.w) - camera.jitter.xy;
    return ViewPixel(vec4<f32>(current, 0.0, 1.0), previous);
}

fn view_detail() -> ViewDetail {
    return ViewDetail(camera.detail.xyz, camera.pixel_angle, camera.texel, camera.pixel_error);
}

fn detail_distance(world: vec3<f32>) -> f32 {
    return length(world - camera.detail.xyz);
}

fn detail_across(world: vec3<f32>) -> f32 {
    return across(view_detail(), world);
}

fn detail_pixels(size: f32, world: vec3<f32>) -> f32 {
    return pixels(view_detail(), size, world);
}

fn detail_fits(error: f32, world: vec3<f32>) -> bool {
    return fits(view_detail(), error, world);
}

fn detail_handover(error: f32, world: vec3<f32>) -> f32 {
    return handover(view_detail(), error, world);
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
