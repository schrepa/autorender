import package::engine::color::{decode_srgb, dithered, encode_srgb};
import package::post::frame::{
    depth_texture, effect_shade, occlusion, occlusion_at, post, reflected, reflections, scene,
};
import package::post::pixel::PostPixel;

struct Resolve {
    texel: vec2<f32>,
    current_weight: f32,
    history_valid: u32,
    options: vec4<f32>,
    render: vec4<f32>,
}

@group(0) @binding(0) var<uniform> taa: Resolve;
@group(0) @binding(2) var motion: texture_2d<f32>;
@group(0) @binding(3) var history: texture_2d<f32>;
@group(0) @binding(4) var history_sampler: sampler;
@group(0) @binding(10) var revealed: texture_2d<f32>;

const STILL_PIXELS: f32 = 0.015625f;

struct Outputs {
    @location(0) canvas: vec4<f32>,
    @location(1) history: vec4<f32>,
}

struct Resolved {
    color: vec3<f32>,
    at: PostPixel,
}

fn ycocg(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * color.r + 0.5 * color.g + 0.25 * color.b,
        0.5 * color.r - 0.5 * color.b,
        -0.25 * color.r + 0.5 * color.g - 0.25 * color.b,
    );
}

fn rgb(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(color.x + color.y - color.z, color.x + color.z, color.x - color.y - color.z);
}

fn sample_history(uv: vec2<f32>) -> vec3<f32> {
    let size = 1.0 / taa.texel;
    let position = uv * size;
    let center = floor(position - 0.5) + 0.5;
    let f = position - center;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let middle = (center + w2 / w12) * taa.texel;
    let before = (center - 1.0) * taa.texel;
    let after = (center + 2.0) * taa.texel;
    let weights = vec4<f32>(w12.x * w0.y, w0.x * w12.y, w3.x * w12.y, w12.x * w3.y);
    let middle_weight = w12.x * w12.y;
    let sum = middle_weight + weights.x + weights.y + weights.z + weights.w;
    let color = textureSampleLevel(history, history_sampler, middle, 0.0).rgb * middle_weight
        + textureSampleLevel(history, history_sampler, vec2<f32>(middle.x, before.y), 0.0).rgb * weights.x
        + textureSampleLevel(history, history_sampler, vec2<f32>(before.x, middle.y), 0.0).rgb * weights.y
        + textureSampleLevel(history, history_sampler, vec2<f32>(after.x, middle.y), 0.0).rgb * weights.z
        + textureSampleLevel(history, history_sampler, vec2<f32>(middle.x, after.y), 0.0).rgb * weights.w;
    return color / sum;
}

fn nearest_motion(pixel: vec2<i32>, last: vec2<i32>) -> vec2<f32> {
    var nearest = pixel;
    var depth = textureLoad(depth_texture, pixel, 0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let at = clamp(pixel + vec2<i32>(x, y), vec2<i32>(0), last);
            let here = textureLoad(depth_texture, at, 0);
            if here > depth {
                depth = here;
                nearest = at;
            }
        }
    }
    return textureLoad(motion, nearest, 0).xy;
}

fn reflected_at(color: vec3<f32>, position: vec2<f32>) -> vec3<f32> {
    return reflected(color, reflections, history_sampler, position / taa.render.xy, post);
}

fn neighbor(pixel: vec2<i32>, offset: vec2<i32>, last: vec2<i32>, shade: f32) -> vec3<f32> {
    let at = clamp(pixel + offset, vec2<i32>(0), last);
    return ycocg(reflected_at(textureLoad(scene, at, 0).rgb * shade, vec2<f32>(at) + 0.5));
}

fn upsampled(position: vec2<f32>, last: vec2<i32>) -> vec3<f32> {
    let base = position - 0.5;
    let first = vec2<i32>(floor(base));
    let f = base - floor(base);
    let a = textureLoad(scene, clamp(first, vec2<i32>(0), last), 0).rgb;
    let b = textureLoad(scene, clamp(first + vec2<i32>(1, 0), vec2<i32>(0), last), 0).rgb;
    let c = textureLoad(scene, clamp(first + vec2<i32>(0, 1), vec2<i32>(0), last), 0).rgb;
    let d = textureLoad(scene, clamp(first + vec2<i32>(1, 1), vec2<i32>(0), last), 0).rgb;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn resolve(position: vec4<f32>) -> Resolved {
    let screen = position.xy * taa.texel;
    let rendered = screen * taa.render.xy;
    let jittered = rendered + taa.render.zw;
    let last = vec2<i32>(taa.render.xy) - 1;
    let pixel = clamp(vec2<i32>(floor(jittered)), vec2<i32>(0), last);
    let texel = textureLoad(scene, pixel, 0);
    let shade = effect_shade(texel.a, occlusion_at(occlusion, pixel, post), post);
    let current = reflected_at(texel.rgb * shade, vec2<f32>(pixel) + 0.5);
    let away = (vec2<f32>(pixel) + 0.5 - taa.render.zw - rendered)
        / (taa.render.xy * taa.texel);
    let near = saturate(2.0 - 2.0 * max(abs(away.x), abs(away.y)));
    var color = current;
    if near < 1.0 {
        color = reflected_at(upsampled(jittered, last) * shade, jittered);
    }
    var offset = textureLoad(motion, pixel, 0).xy;
    if taa.options.y != 0.0 {
        offset = nearest_motion(pixel, last);
    }
    let uv = screen + offset;
    if taa.history_valid != 0u && all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)) {
        let center = ycocg(current);
        let left = neighbor(pixel, vec2<i32>(-1, 0), last, shade);
        let right = neighbor(pixel, vec2<i32>(1, 0), last, shade);
        let up = neighbor(pixel, vec2<i32>(0, -1), last, shade);
        let down = neighbor(pixel, vec2<i32>(0, 1), last, shade);
        var low = min(center, min(min(left, right), min(up, down)));
        var high = max(center, max(max(left, right), max(up, down)));
        if taa.options.z != 0.0 {
            let a = neighbor(pixel, vec2<i32>(-1, -1), last, shade);
            let b = neighbor(pixel, vec2<i32>(1, -1), last, shade);
            let c = neighbor(pixel, vec2<i32>(-1, 1), last, shade);
            let d = neighbor(pixel, vec2<i32>(1, 1), last, shade);
            low = 0.5 * (low + min(low, min(min(a, b), min(c, d))));
            high = 0.5 * (high + max(high, max(max(a, b), max(c, d))));
        }
        var past: vec3<f32>;
        if all(abs(offset) < taa.texel * STILL_PIXELS) {
            past = textureLoad(history, vec2<i32>(position.xy), 0).rgb;
        } else {
            past = sample_history(uv);
        }
        var weight = taa.current_weight * near;
        if taa.options.w > 0.0 {
            let covered = min(pixel, vec2<i32>(textureDimensions(revealed)) - 1);
            let particles = 1.0 - textureLoad(revealed, covered, 0).y;
            weight = max(weight, min(taa.options.w * particles, 1.0));
        }
        color = mix(rgb(clamp(ycocg(past), low, high)), current, weight);
    }
    color = max(color, vec3<f32>(0.0));
    return Resolved(color, PostPixel(position.xy, screen, pixel, taa.options.x, 2u));
}

fn output(resolved: Resolved, encoded: vec3<f32>) -> Outputs {
    var outputs: Outputs;
    outputs.history = vec4<f32>(resolved.color, 1.0);
    if post.sun_direction.w != 0.0 {
        outputs.history = vec4<f32>(decode_srgb(dithered(encode_srgb(resolved.color),
            resolved.at.position, resolved.at.turn, 1u)), 1.0);
    }
    outputs.canvas = vec4<f32>(encoded, 1.0);
    return outputs;
}
