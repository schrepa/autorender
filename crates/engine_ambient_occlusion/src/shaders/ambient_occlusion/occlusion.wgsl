import package::engine::math::TAU;
import package::engine::screen_points::{depth_normal, occluder_at, world_at};
import package::engine::post::frame::post;
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

fn occlusion_share(pixels: f32) -> f32 {
    let handover = piece().canopy.x * post.eye.w;
    return saturate(log2(max(pixels, 1.0e-6) / max(handover, 1.0e-6)));
}

fn ambient_visibility(pixel: vec2<i32>, p: vec3<f32>, normal: vec3<f32>, distance: f32,
        noise: f32) -> f32 {
    if piece().occlusion.w == 0.0 {
        return 1.0;
    }
    let last = vec2<i32>(post.size.xy) - 1;
    let radius = occlusion_radius();
    let seen = radius * post.eye.w / max(length(p - post.detail.xyz), 1.0e-4);
    let strength = piece().occlusion.y * occlusion_share(seen);
    let pixels = min(radius * post.eye.w / distance, MAX_PIXELS);
    if strength <= 0.0 || pixels < 1.0 {
        return 1.0;
    }
    let turn = vec2<f32>(cos(noise * TAU), sin(noise * TAU));
    let radius2 = radius * radius;
    let bias = 0.002 * distance;
    var sum = 0.0;
    for (var i = 0u; i < SAMPLES; i++) {
        let fraction = (f32(i) + 0.5) / f32(SAMPLES);
        let d = SPIRAL[i];
        let offset = vec2<f32>(d.x * turn.x - d.y * turn.y, d.x * turn.y + d.y * turn.x)
            * (fraction * pixels);
        let probe = occluder_at(clamp(pixel + vec2<i32>(round(offset)), vec2<i32>(0), last));
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
    let normal = depth_normal(pixel, p);
    if all(normal == vec3<f32>(0.0)) {
        return vec2<f32>(1.0, distance);
    }
    let noise = fract(52.9829189 * fract(dot(position.xy, vec2<f32>(0.06711056, 0.00583715))));
    return vec2<f32>(ambient_visibility(pixel, p, normal, distance, noise), distance);
}

const OCCLUSION_PLANE: f32 = 0.01;

fn inverse_distance(sample: vec4<f32>) -> f32 {
    return select(0.0, 1.0 / sample.y, sample.y > 0.0);
}

fn filtered_occlusion(pixel: vec2<i32>) -> f32 {
    let last = vec2<i32>(post.size.xy) - 1;
    let center = textureLoad(raw_occlusion, pixel, 0);
    if !(center.y > 0.0) {
        return center.x;
    }
    var samples: array<vec4<f32>, 25>;
    var inverses: array<f32, 25>;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let index = (y + 2) * 5 + x + 2;
            var sample = center;
            if x != 0 || y != 0 {
                sample = textureLoad(raw_occlusion, clamp(pixel + vec2<i32>(x, y), vec2<i32>(0),
                    last), 0);
            }
            samples[index] = sample;
            inverses[index] = inverse_distance(sample);
        }
    }
    let inverse = inverses[12];
    let right = inverses[13] - inverse;
    let left = inverse - inverses[11];
    let down = inverses[17] - inverse;
    let up = inverse - inverses[7];
    let slope = vec2<f32>(
        select(right, left, abs(left) < abs(right)),
        select(down, up, abs(up) < abs(down)),
    );
    let tolerance = OCCLUSION_PLANE * inverse;
    var sum = 0.0;
    var total = 0.0;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let index = (y + 2) * 5 + x + 2;
            let offset = vec2<i32>(x, y);
            let expected = inverse + dot(slope, vec2<f32>(offset));
            let plane = saturate(1.25 - abs(inverses[index] - expected) / tolerance);
            let weight = exp(-f32(x * x + y * y) / 8.0) * plane;
            sum += samples[index].x * weight;
            total += weight;
        }
    }
    return sum / total;
}

@fragment
fn fs_effects_filter(@builtin(position) position: vec4<f32>) -> @location(0) f32 {
    return filtered_occlusion(vec2<i32>(position.xy));
}
