import package::engine::math::TAU;
import package::engine::screen_points::{depth_normal, world_at};
import package::engine::post::frame::{depth_texture, post};
import package::generated::pieces::ambient_occlusion::piece;

@group(1) @binding(0) var raw_occlusion: texture_2d<f32>;

const SAMPLES: u32 = 8u;
const SPIRAL: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2<f32>(-0.9238795, 0.38268343), vec2<f32>(-0.38268343, 0.9238795),
    vec2<f32>(0.38268343, 0.9238795), vec2<f32>(0.9238795, 0.38268343),
    vec2<f32>(0.9238795, -0.38268343), vec2<f32>(0.38268343, -0.9238795),
    vec2<f32>(-0.38268343, -0.9238795), vec2<f32>(-0.9238795, -0.38268343),
);
const MAX_PIXELS: f32 = 96.0;

fn occlusion_radius() -> f32 {
    return piece().occlusion.x;
}

fn occluding_pixels(distance: f32) -> f32 {
    let pixels = min(occlusion_radius() * post.eye.w / distance, MAX_PIXELS);
    if piece().occlusion.w == 0.0 || piece().occlusion.y <= 0.0 || pixels < 1.0 {
        return 0.0;
    }
    return pixels;
}

fn ambient_visibility(pixel: vec2<i32>, p: vec3<f32>, normal: vec3<f32>, distance: f32,
        pixels: f32, noise: f32) -> f32 {
    let last = vec2<i32>(post.size.xy) - 1;
    let radius = occlusion_radius();
    let strength = piece().occlusion.y;
    let turn = vec2<f32>(cos(noise * TAU), sin(noise * TAU));
    let radius2 = radius * radius;
    let bias = 0.002 * distance;
    var sum = 0.0;
    for (var i = 0u; i < SAMPLES; i++) {
        let fraction = (f32(i) + 0.5) / f32(SAMPLES);
        let d = SPIRAL[i];
        let offset = vec2<f32>(d.x * turn.x - d.y * turn.y, d.x * turn.y + d.y * turn.x)
            * (fraction * pixels);
        let probe = world_at(clamp(pixel + vec2<i32>(round(offset)), vec2<i32>(0), last));
        let v = probe.xyz - p;
        let vv = dot(v, v);
        let falloff = max(radius2 - vv, 0.0);
        sum += probe.w * falloff * falloff * falloff
            * max((dot(v, normal) - bias) / (vv + 0.01 * radius2), 0.0);
    }
    let radius6 = radius2 * radius2 * radius2;
    let obscurance = clamp(sum * (5.0 / f32(SAMPLES)) / radius6, 0.0, 1.0);
    return 1.0 - strength * obscurance;
}

@fragment
fn fs_occlusion(@builtin(position) position: vec4<f32>) -> @location(0) vec2<f32> {
    let pixel = vec2<i32>(position.xy);
    let center = world_at(pixel);
    if center.w == 0.0 {
        return vec2<f32>(1.0, 0.0);
    }
    let p = center.xyz;
    let distance = length(post.eye.xyz - p);
    let pixels = occluding_pixels(distance);
    if pixels == 0.0 {
        return vec2<f32>(1.0, distance);
    }
    let normal = depth_normal(pixel, p);
    if all(normal == vec3<f32>(0.0)) {
        return vec2<f32>(1.0, distance);
    }
    let noise = fract(52.9829189 * fract(dot(position.xy, vec2<f32>(0.06711056, 0.00583715))));
    return vec2<f32>(ambient_visibility(pixel, p, normal, distance, pixels, noise), distance);
}

const OCCLUSION_PLANE: f32 = 0.01;

fn inverse_distance(sample: vec4<f32>) -> f32 {
    return select(0.0, 1.0 / sample.y, sample.y > 0.0);
}

fn filter_tap(center: vec4<f32>, pixel: vec2<i32>, offset: vec2<i32>, last: vec2<i32>)
        -> vec4<f32> {
    if all(offset == vec2<i32>(0)) {
        return center;
    }
    return textureLoad(raw_occlusion, clamp(pixel + offset, vec2<i32>(0), last), 0);
}

struct FilterFit {
    inverse: f32,
    slope: vec2<f32>,
    tolerance: f32,
}

fn filter_add(held: vec2<f32>, center: vec4<f32>, pixel: vec2<i32>, offset: vec2<i32>,
        last: vec2<i32>, fit: FilterFit) -> vec2<f32> {
    let sample = filter_tap(center, pixel, offset, last);
    let expected = fit.inverse + dot(fit.slope, vec2<f32>(offset));
    let plane = saturate(1.25 - abs(inverse_distance(sample) - expected) / fit.tolerance);
    let weight = exp(-f32(offset.x * offset.x + offset.y * offset.y) / 8.0) * plane;
    return vec2<f32>(held.x + sample.x * weight, held.y + weight);
}

fn filter_row(held: vec2<f32>, center: vec4<f32>, pixel: vec2<i32>, y: i32, last: vec2<i32>,
        fit: FilterFit) -> vec2<f32> {
    var row = filter_add(held, center, pixel, vec2<i32>(-2, y), last, fit);
    row = filter_add(row, center, pixel, vec2<i32>(-1, y), last, fit);
    row = filter_add(row, center, pixel, vec2<i32>(0, y), last, fit);
    row = filter_add(row, center, pixel, vec2<i32>(1, y), last, fit);
    return filter_add(row, center, pixel, vec2<i32>(2, y), last, fit);
}

fn filtered_occlusion(pixel: vec2<i32>) -> f32 {
    let last = vec2<i32>(post.size.xy) - 1;
    let center = textureLoad(raw_occlusion, pixel, 0);
    if !(center.y > 0.0) {
        return center.x;
    }
    let inverse = inverse_distance(center);
    let right = inverse_distance(filter_tap(center, pixel, vec2<i32>(1, 0), last)) - inverse;
    let left = inverse - inverse_distance(filter_tap(center, pixel, vec2<i32>(-1, 0), last));
    let down = inverse_distance(filter_tap(center, pixel, vec2<i32>(0, 1), last)) - inverse;
    let up = inverse - inverse_distance(filter_tap(center, pixel, vec2<i32>(0, -1), last));
    let slope = vec2<f32>(
        select(right, left, abs(left) < abs(right)),
        select(down, up, abs(up) < abs(down)),
    );
    let fit = FilterFit(inverse, slope, OCCLUSION_PLANE * inverse);
    var held = filter_row(vec2<f32>(0.0), center, pixel, -2, last, fit);
    held = filter_row(held, center, pixel, -1, last, fit);
    held = filter_row(held, center, pixel, 0, last, fit);
    held = filter_row(held, center, pixel, 1, last, fit);
    held = filter_row(held, center, pixel, 2, last, fit);
    return held.x / held.y;
}

@fragment
fn fs_effects_filter(@builtin(position) position: vec4<f32>) -> @location(0) vec2<f32> {
    let pixel = vec2<i32>(position.xy);
    return vec2<f32>(1.0 - filtered_occlusion(pixel),
        quantizeToF16(textureLoad(depth_texture, pixel, 0)));
}
