import package::engine::post::frame::{drawn_uv, linear, post, scene};
import package::engine::post::pixel::PostPixel;

fn upscaled_scene(pixel: vec2<f32>) -> vec3<f32> {
    let center = floor(pixel - 0.5) + 0.5;
    let f = pixel - center;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let middle = center + w2 / w12;
    let before = center - 1.0;
    let after = center + 2.0;
    let weights = vec4<f32>(w12.x * w0.y, w0.x * w12.y, w3.x * w12.y, w12.x * w3.y);
    let middle_weight = w12.x * w12.y;
    let sum = middle_weight + weights.x + weights.y + weights.z + weights.w;
    let color = scene_tap(middle) * middle_weight
        + scene_tap(vec2<f32>(middle.x, before.y)) * weights.x
        + scene_tap(vec2<f32>(before.x, middle.y)) * weights.y
        + scene_tap(vec2<f32>(after.x, middle.y)) * weights.z
        + scene_tap(vec2<f32>(middle.x, after.y)) * weights.w;
    return max(color / sum, vec3<f32>(0.0));
}

fn scene_tap(pixel: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(scene, linear, drawn_uv(pixel, post), 0.0).rgb;
}

fn post_pixel(position: vec4<f32>) -> PostPixel {
    let uv = position.xy * post.extent.zw;
    let whole = all(post.size.xy == post.extent.xy);
    let rendered = select(uv * post.size.xy, position.xy, whole);
    let pixel = min(vec2<i32>(rendered), vec2<i32>(post.size.xy) - 1);
    return PostPixel(position.xy, uv, pixel);
}

fn scene_color(at: PostPixel) -> vec4<f32> {
    if any(post.size.xy != post.extent.xy) {
        return vec4<f32>(upscaled_scene(at.uv * post.size.xy), 1.0);
    }
    return vec4<f32>(textureLoad(scene, at.pixel, 0).rgb, 1.0);
}
