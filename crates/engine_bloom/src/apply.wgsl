import package::post::frame::{Post, drawn_texels, post};
import package::post::pixel::PostPixel;

@group(0) @binding(6) var bloom: texture_2d<f32>;
@group(0) @binding(20) var bloom_sampler: sampler;

fn glowing(color: vec3<f32>, bloom: texture_2d<f32>, linear: sampler, uv: vec2<f32>,
        settings: Post) -> vec3<f32> {
    if settings.bloom.y == 0.0 {
        return color;
    }
    let glow = max(textureSampleLevel(bloom, linear, uv, 0.0).rgb * settings.bloom.x,
        vec3<f32>(0.0));
    let light = glow / (vec3<f32>(1.0) + glow);
    return vec3<f32>(1.0) - (vec3<f32>(1.0) - color) * (vec3<f32>(1.0) - light);
}

fn apply(color: vec3<f32>, at: PostPixel) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(bloom));
    let drawn = drawn_texels(size, post);
    return glowing(color, bloom, bloom_sampler, clamp(at.uv * drawn, vec2<f32>(0.5), drawn - 0.5)
        / size, post);
}
