import package::engine::post::frame::{depth_texture, linear, post};
import package::engine::transparency::cover;
import package::face::tonemap::exposure;
import package::generated::pieces::taa::piece;
import package::generated::taa::TaaView;

@group(1) @binding(0) var scene_color: texture_2d<f32>;
@group(1) @binding(1) var history: texture_2d<f32>;
@group(1) @binding(2) var<uniform> taa_view: TaaView;
@group(1) @binding(3) var motion_vectors: texture_2d<f32>;
@group(1) @binding(4) var blended_mask: texture_2d<f32>;

const STILL_PIXELS: f32 = 0.015625f;

struct Centre {
    screen: vec2<f32>,
    rendered: vec2<f32>,
    jittered: vec2<f32>,
    last: vec2<i32>,
    pixel: vec2<i32>,
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

fn weighed(color: vec3<f32>) -> vec3<f32> {
    return color / (1.0 + max(color.r, max(color.g, color.b)) * exposure());
}

fn unweighed(color: vec3<f32>) -> vec3<f32> {
    return color / max(1.0 - max(color.r, max(color.g, color.b)) * exposure(), 1.0e-6);
}

fn sample_history(uv: vec2<f32>) -> vec3<f32> {
    let texel = post.extent.zw;
    let size = 1.0 / texel;
    let position = uv * size;
    let center = floor(position - 0.5) + 0.5;
    let f = position - center;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let middle = (center + w2 / w12) * texel;
    let before = (center - 1.0) * texel;
    let after = (center + 2.0) * texel;
    let weights = vec4<f32>(w12.x * w0.y, w0.x * w12.y, w3.x * w12.y, w12.x * w3.y);
    let middle_weight = w12.x * w12.y;
    let sum = middle_weight + weights.x + weights.y + weights.z + weights.w;
    let color = textureSampleLevel(history, linear, middle, 0.0).rgb * middle_weight
        + textureSampleLevel(history, linear, vec2<f32>(middle.x, before.y), 0.0).rgb * weights.x
        + textureSampleLevel(history, linear, vec2<f32>(before.x, middle.y), 0.0).rgb * weights.y
        + textureSampleLevel(history, linear, vec2<f32>(after.x, middle.y), 0.0).rgb * weights.z
        + textureSampleLevel(history, linear, vec2<f32>(middle.x, after.y), 0.0).rgb * weights.w;
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
    return textureLoad(motion_vectors, nearest, 0).xy;
}

fn neighbor(pixel: vec2<i32>, offset: vec2<i32>, last: vec2<i32>) -> vec3<f32> {
    let at = clamp(pixel + offset, vec2<i32>(0), last);
    return ycocg(weighed(textureLoad(scene_color, at, 0).rgb));
}

fn upsampled(position: vec2<f32>, last: vec2<i32>) -> vec3<f32> {
    let base = position - 0.5;
    let first = vec2<i32>(floor(base));
    let f = base - floor(base);
    let a = textureLoad(scene_color, clamp(first, vec2<i32>(0), last), 0).rgb;
    let b = textureLoad(scene_color, clamp(first + vec2<i32>(1, 0), vec2<i32>(0), last), 0).rgb;
    let c = textureLoad(scene_color, clamp(first + vec2<i32>(0, 1), vec2<i32>(0), last), 0).rgb;
    let d = textureLoad(scene_color, clamp(first + vec2<i32>(1, 1), vec2<i32>(0), last), 0).rgb;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn centre(position: vec2<f32>) -> Centre {
    let screen = position * post.extent.zw;
    let rendered = screen * post.size.xy;
    let jittered = rendered + taa_view.offset;
    let last = vec2<i32>(post.size.xy) - 1;
    let pixel = clamp(vec2<i32>(floor(jittered)), vec2<i32>(0), last);
    return Centre(screen, rendered, jittered, last, pixel);
}

@fragment
fn fs_temporal_resolve(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let tap = centre(position.xy);
    let screen = tap.screen;
    let rendered = tap.rendered;
    let jittered = tap.jittered;
    let last = tap.last;
    let pixel = tap.pixel;
    let current = weighed(textureLoad(scene_color, pixel, 0).rgb);
    let away = (vec2<f32>(pixel) + 0.5 - taa_view.offset - rendered)
        / (post.size.xy * post.extent.zw);
    let near = saturate(2.0 - 2.0 * max(abs(away.x), abs(away.y)));
    var color = current;
    if near < 1.0 {
        color = weighed(upsampled(jittered, last));
    }
    var offset = textureLoad(motion_vectors, pixel, 0).xy;
    if piece().options.x != 0.0 {
        offset = nearest_motion(pixel, last);
    }
    let uv = screen + offset;
    if taa_view.history != 0u && all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)) {
        let center = ycocg(current);
        let left = neighbor(pixel, vec2<i32>(-1, 0), last);
        let right = neighbor(pixel, vec2<i32>(1, 0), last);
        let up = neighbor(pixel, vec2<i32>(0, -1), last);
        let down = neighbor(pixel, vec2<i32>(0, 1), last);
        var low = min(center, min(min(left, right), min(up, down)));
        var high = max(center, max(max(left, right), max(up, down)));
        if piece().options.y != 0.0 {
            let a = neighbor(pixel, vec2<i32>(-1, -1), last);
            let b = neighbor(pixel, vec2<i32>(1, -1), last);
            let c = neighbor(pixel, vec2<i32>(-1, 1), last);
            let d = neighbor(pixel, vec2<i32>(1, 1), last);
            low = 0.5 * (low + min(low, min(min(a, b), min(c, d))));
            high = 0.5 * (high + max(high, max(max(a, b), max(c, d))));
        }
        var past: vec3<f32>;
        if all(abs(offset) < post.extent.zw * STILL_PIXELS) {
            past = textureLoad(history, vec2<i32>(position.xy), 0).rgb;
        } else {
            past = sample_history(uv);
        }
        past = weighed(max(past, vec3<f32>(0.0)));
        var weight = piece().options.w * near;
        if piece().options.z > 0.0 {
            let covered = cover(blended_mask, pixel);
            weight = max(weight, min(piece().options.z * covered, 1.0));
        }
        color = mix(rgb(clamp(ycocg(past), low, high)), current, weight);
    }
    return vec4<f32>(unweighed(max(color, vec3<f32>(0.0))), 1.0);
}
