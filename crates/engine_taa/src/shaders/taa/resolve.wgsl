import package::engine::post::frame::{depth_texture, linear, post, scene};
import package::engine::post::pixel::{PostPixel, SceneLight};
import package::face::dither::dithering;
import package::face::post_scene::apply as scene_light;
import package::face::transparency::revealed;
import package::generated::pieces::taa::piece;
import package::generated::targets::{output_history, output_motion};

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
    let color = textureSampleLevel(output_history, linear, middle, 0.0).rgb * middle_weight
        + textureSampleLevel(output_history, linear, vec2<f32>(middle.x, before.y), 0.0).rgb * weights.x
        + textureSampleLevel(output_history, linear, vec2<f32>(before.x, middle.y), 0.0).rgb * weights.y
        + textureSampleLevel(output_history, linear, vec2<f32>(after.x, middle.y), 0.0).rgb * weights.z
        + textureSampleLevel(output_history, linear, vec2<f32>(middle.x, after.y), 0.0).rgb * weights.w;
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
    return textureLoad(output_motion, nearest, 0).xy;
}

fn turn() -> f32 {
    return select(0.0, piece().options.x, dithering());
}

fn taa_at(position: vec2<f32>, pixel: vec2<i32>) -> PostPixel {
    return PostPixel(position, position / post.size.xy, pixel, turn(), 2u);
}

fn neighbor(pixel: vec2<i32>, offset: vec2<i32>, last: vec2<i32>, ambient: f32) -> vec3<f32> {
    let at = clamp(pixel + offset, vec2<i32>(0), last);
    let texel = textureLoad(scene, at, 0);
    let light = SceneLight(texel.rgb, ambient);
    return ycocg(scene_light(light, taa_at(vec2<f32>(at) + 0.5, pixel)).color);
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

fn centre(position: vec2<f32>) -> Centre {
    let screen = position * post.extent.zw;
    let rendered = screen * post.size.xy;
    let jittered = rendered + piece().offset;
    let last = vec2<i32>(post.size.xy) - 1;
    let pixel = clamp(vec2<i32>(floor(jittered)), vec2<i32>(0), last);
    return Centre(screen, rendered, jittered, last, pixel);
}

fn resolve(at: PostPixel, light: SceneLight) -> SceneLight {
    let tap = centre(at.position);
    let screen = tap.screen;
    let rendered = tap.rendered;
    let jittered = tap.jittered;
    let last = tap.last;
    let pixel = tap.pixel;
    let texel = textureLoad(scene, pixel, 0);
    let current = scene_light(SceneLight(texel.rgb, texel.a),
        taa_at(vec2<f32>(pixel) + 0.5, pixel)).color;
    let away = (vec2<f32>(pixel) + 0.5 - piece().offset - rendered)
        / (post.size.xy * post.extent.zw);
    let near = saturate(2.0 - 2.0 * max(abs(away.x), abs(away.y)));
    var color = current;
    if near < 1.0 {
        color = scene_light(SceneLight(upsampled(jittered, last), texel.a),
            taa_at(jittered, pixel)).color;
    }
    var offset = textureLoad(output_motion, pixel, 0).xy;
    if piece().options.y != 0.0 {
        offset = nearest_motion(pixel, last);
    }
    let uv = screen + offset;
    if piece().history_valid != 0u && all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)) {
        let center = ycocg(current);
        let left = neighbor(pixel, vec2<i32>(-1, 0), last, texel.a);
        let right = neighbor(pixel, vec2<i32>(1, 0), last, texel.a);
        let up = neighbor(pixel, vec2<i32>(0, -1), last, texel.a);
        let down = neighbor(pixel, vec2<i32>(0, 1), last, texel.a);
        var low = min(center, min(min(left, right), min(up, down)));
        var high = max(center, max(max(left, right), max(up, down)));
        if piece().options.z != 0.0 {
            let a = neighbor(pixel, vec2<i32>(-1, -1), last, texel.a);
            let b = neighbor(pixel, vec2<i32>(1, -1), last, texel.a);
            let c = neighbor(pixel, vec2<i32>(-1, 1), last, texel.a);
            let d = neighbor(pixel, vec2<i32>(1, 1), last, texel.a);
            low = 0.5 * (low + min(low, min(min(a, b), min(c, d))));
            high = 0.5 * (high + max(high, max(max(a, b), max(c, d))));
        }
        var past: vec3<f32>;
        if all(abs(offset) < post.extent.zw * STILL_PIXELS) {
            past = textureLoad(output_history, vec2<i32>(at.position), 0).rgb;
        } else {
            past = sample_history(uv);
        }
        var weight = piece().current_weight * near;
        if piece().options.w > 0.0 {
            let particles = 1.0 - revealed(pixel).y;
            weight = max(weight, min(piece().options.w * particles, 1.0));
        }
        color = mix(rgb(clamp(ycocg(past), low, high)), current, weight);
    }
    color = max(color, vec3<f32>(0.0));
    return SceneLight(color, light.ambient);
}

fn resolved_at(at: PostPixel) -> PostPixel {
    let tap = centre(at.position);
    return PostPixel(at.position, tap.screen, tap.pixel, turn(), 2u);
}
