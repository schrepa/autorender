import package::engine::color::luminance;
import package::engine::screen_points::{depth_normal, world_at};
import package::post::frame::{drawn_texels, drawn_uv, glow, linear, post, scene};
import package::post::haze::haze;

@group(0) @binding(4) var source: texture_2d<f32>;
@group(0) @binding(9) var marched: texture_2d<f32>;
@group(0) @binding(12) var raw_occlusion: texture_2d<f32>;
@group(0) @binding(15) var bloom_source: texture_2d<f32>;

const SKY_DIRECTIONS: u32 = 13u;
const SKY_LAYERS: u32 = 26u;
struct Sky {
    clips: array<mat4x4<f32>, SKY_LAYERS>,
    directions: array<vec4<f32>, SKY_LAYERS>,
    counts: vec4<f32>,
}
@group(0) @binding(16) var sky_maps: texture_depth_2d_array;
@group(0) @binding(17) var sky_sampler: sampler_comparison;
@group(0) @binding(18) var<uniform> sky: Sky;

const TAU: f32 = 6.2831853;
const SAMPLES: u32 = 8u;
const SPIRAL: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2<f32>(-0.9238795, 0.38268343), vec2<f32>(-0.38268343, 0.9238795),
    vec2<f32>(0.38268343, 0.9238795), vec2<f32>(0.9238795, 0.38268343),
    vec2<f32>(0.9238795, -0.38268343), vec2<f32>(0.38268343, -0.9238795),
    vec2<f32>(-0.38268343, -0.9238795), vec2<f32>(-0.9238795, -0.38268343),
);
const MAX_PIXELS: f32 = 96.0;

fn occlusion_radius() -> f32 {
    return post.occlusion.x;
}

fn occlusion_share(pixels: f32) -> f32 {
    let handover = post.occlusion.z * post.eye.w;
    return saturate(log2(max(pixels, 1.0e-6) / max(handover, 1.0e-6)));
}

fn ambient_visibility(pixel: vec2<i32>, p: vec3<f32>, normal: vec3<f32>, distance: f32,
        noise: f32) -> f32 {
    if post.occlusion.w == 0.0 {
        return 1.0;
    }
    let last = vec2<i32>(post.size.xy) - 1;
    let radius = occlusion_radius();
    let seen = radius * post.eye.w / max(length(p - post.detail.xyz), 1.0e-4);
    let strength = post.occlusion.y * occlusion_share(seen);
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

fn sky_seen(layer: u32, p: vec3<f32>, normal: vec3<f32>, texel: f32) -> vec2<f32> {
    let toward = sky.directions[layer];
    if toward.w <= 0.0 {
        return vec2<f32>(1.0, 0.0);
    }
    let lifted = p + normal * (1.5 * texel) + toward.xyz * texel;
    let clip = sky.clips[layer] * vec4<f32>(lifted, 1.0);
    let edge = max(abs(clip.x), abs(clip.y));
    let counted = toward.w * (1.0 - smoothstep(0.85, 1.0, edge)) * select(1.0, 0.0, clip.z >= 1.0);
    if counted <= 0.0 {
        return vec2<f32>(1.0, 0.0);
    }
    let uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
    let seen = textureSampleCompareLevel(sky_maps, sky_sampler, uv, layer, clip.z);
    return vec2<f32>(seen, counted);
}

fn sky_visibility(p: vec3<f32>, normal: vec3<f32>) -> f32 {
    let count = u32(sky.counts.x);
    if post.sky.w == 0.0 || count == 0u {
        return 1.0;
    }
    var sum = 0.0;
    var total = 0.0;
    for (var direction = 0u; direction < count; direction++) {
        let weight = max(dot(normal, sky.directions[direction].xyz), 0.0) + 0.05;
        let near = sky_seen(direction, p, normal, sky.counts.y);
        var seen = near.x;
        if near.y < 1.0 {
            let far = sky_seen(direction + SKY_DIRECTIONS, p, normal, sky.counts.z);
            seen = mix(mix(1.0, far.x, far.y), near.x, near.y);
        }
        sum += weight * seen;
        total += weight;
    }
    let visible = max(sum / total, post.sky.y);
    return 1.0 - post.sky.x * (1.0 - visible);
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
    var visible = sky_visibility(p, normal);
    if post.occlusion.w != 0.0 {
        visible *= ambient_visibility(pixel, p, normal, distance, noise);
    }
    return vec2<f32>(visible, distance);
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

const HAZE_CURRENT: f32 = 0.1;

const HAZE_PLANE: f32 = 0.1;

fn reprojected_haze(pixel: vec2<i32>, own: f32) -> vec4<f32> {
    let drawn = post.haze.yz;
    if drawn.x <= 0.0 {
        return vec4<f32>(0.0);
    }
    let point = world_at(pixel);
    let clip = post.previous_view_projection * vec4<f32>(point.xyz, 1.0);
    if !(clip.w > 0.0) {
        return vec4<f32>(0.0);
    }
    let ndc = clip.xy / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return vec4<f32>(0.0);
    }
    var expected = 0.0;
    if own > 0.0 {
        expected = 1.0 / max(length(point.xyz - post.previous_eye.xyz), 1.0e-4);
    }
    let tolerance = max(HAZE_PLANE * expected, 1.0e-7);
    let at = uv * drawn - 0.5;
    let first = vec2<i32>(floor(at));
    let f = at - floor(at);
    let last = vec2<i32>(drawn) - 1;
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let tap = textureLoad(haze, clamp(first + vec2<i32>(x, y), vec2<i32>(0), last), 0);
            let bilinear = select(1.0 - f.x, f.x, x == 1) * select(1.0 - f.y, f.y, y == 1);
            let weight = bilinear * saturate(1.25 - abs(tap.w - expected) / tolerance);
            sum += tap.rgb * weight;
            total += weight;
        }
    }
    if total < 1.0e-3 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(sum / total, total);
}

@fragment
fn fs_haze_filter(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let last = vec2<i32>(post.size.xy) - 1;
    let center = textureLoad(marched, pixel, 0);
    let own = center.w;
    let tolerance = max(HAZE_PLANE * own, 1.0e-7);
    var low = center.rgb;
    var high = center.rgb;
    var surface = center.rgb;
    var count = 1.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            if x == 0 && y == 0 {
                continue;
            }
            let neighbor = textureLoad(marched,
                clamp(pixel + vec2<i32>(x, y), vec2<i32>(0), last), 0);
            low = min(low, neighbor.rgb);
            high = max(high, neighbor.rgb);
            let same = saturate(1.25 - abs(neighbor.w - own) / tolerance);
            surface += neighbor.rgb * same;
            count += same;
        }
    }
    let history = reprojected_haze(pixel, own);
    if history.w <= 0.0 {
        return vec4<f32>(surface / count, own);
    }
    return vec4<f32>(mix(clamp(history.rgb, low, high), center.rgb, HAZE_CURRENT), own);
}

fn level_tap(level: texture_2d<f32>, at: vec2<f32>, drawn: vec2<f32>, size: vec2<f32>)
        -> vec3<f32> {
    return textureSampleLevel(level, linear, clamp(at, vec2<f32>(0.5), drawn - 0.5) / size,
        0.0).rgb;
}

fn glow_light(uv: vec2<f32>) -> vec3<f32> {
    let color = textureSampleLevel(scene, linear, uv, 0.0).rgb;
    let weight = min(textureSampleLevel(glow, linear, uv, 0.0).r, 0.98);
    return color * (weight / (1.0 - weight));
}

@fragment
fn fs_bloom_first(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let center = position.xy * 2.0;
    var sum = vec3<f32>(0.0);
    var weights = 0.0;
    for (var i = 0u; i < 5u; i++) {
        var offset = vec2<f32>(0.0);
        if i > 0u {
            offset = vec2<f32>(f32(i & 1u) * 2.0 - 1.0, f32((i >> 1u) & 1u) * 2.0 - 1.0);
        }
        let light = glow_light(drawn_uv(center + offset, post));
        let weight = select(1.0, 4.0, i == 0u) / (1.0 + luminance(light));
        sum += light * weight;
        weights += weight;
    }
    return vec4<f32>(sum / weights, 1.0);
}

fn halved(level: texture_2d<f32>, position: vec2<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(level));
    let drawn = drawn_texels(size, post);
    let at = position * 2.0;
    let sum = level_tap(level, at, drawn, size) * 4.0
        + level_tap(level, at - 1.0, drawn, size)
        + level_tap(level, at + 1.0, drawn, size)
        + level_tap(level, at + vec2<f32>(1.0, -1.0), drawn, size)
        + level_tap(level, at + vec2<f32>(-1.0, 1.0), drawn, size);
    return vec4<f32>(sum * 0.125, 1.0);
}

@fragment
fn fs_bloom_down(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return halved(source, position.xy);
}

struct Levels {
    @location(0) scene: vec4<f32>,
    @location(1) bloom: vec4<f32>,
}

@fragment
fn fs_levels_first(@builtin(position) position: vec4<f32>) -> Levels {
    let center = position.xy * 2.0;
    var colors: array<vec3<f32>, 5>;
    var sum = vec3<f32>(0.0);
    var weights = 0.0;
    for (var i = 0u; i < 5u; i++) {
        var offset = vec2<f32>(0.0);
        if i > 0u {
            offset = vec2<f32>(f32(i & 1u) * 2.0 - 1.0, f32((i >> 1u) & 1u) * 2.0 - 1.0);
        }
        let at = drawn_uv(center + offset, post);
        let color = textureSampleLevel(scene, linear, at, 0.0).rgb;
        let decoded = min(textureSampleLevel(glow, linear, at, 0.0).r, 0.98);
        let light = color * (decoded / (1.0 - decoded));
        let weight = select(1.0, 4.0, i == 0u) / (1.0 + luminance(light));
        sum += light * weight;
        weights += weight;
        colors[i] = color;
    }
    var levels: Levels;
    levels.scene = vec4<f32>(
        (colors[0] * 4.0 + colors[4] + colors[3] + colors[1] + colors[2]) * 0.125, 1.0);
    levels.bloom = vec4<f32>(sum / weights, 1.0);
    return levels;
}

@fragment
fn fs_levels_down(@builtin(position) position: vec4<f32>) -> Levels {
    var levels: Levels;
    levels.scene = halved(source, position.xy);
    levels.bloom = halved(bloom_source, position.xy);
    return levels;
}

@fragment
fn fs_bloom_up(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(source));
    let drawn = drawn_texels(size, post);
    let at = position.xy * 0.5;
    let sum = (level_tap(source, at + vec2<f32>(-2.0, 0.0), drawn, size)
        + level_tap(source, at + vec2<f32>(2.0, 0.0), drawn, size)
        + level_tap(source, at + vec2<f32>(0.0, -2.0), drawn, size)
        + level_tap(source, at + vec2<f32>(0.0, 2.0), drawn, size))
        + (level_tap(source, at - 1.0, drawn, size)
        + level_tap(source, at + 1.0, drawn, size)
        + level_tap(source, at + vec2<f32>(1.0, -1.0), drawn, size)
        + level_tap(source, at + vec2<f32>(-1.0, 1.0), drawn, size)) * 2.0;
    return vec4<f32>(sum / 12.0, 1.0);
}
