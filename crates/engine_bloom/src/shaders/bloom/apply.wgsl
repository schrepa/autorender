import package::engine::post::frame::{drawn_texels, post};
import package::engine::post::pixel::PostPixel;
import package::generated::pieces::bloom::piece;

@group(0) @binding(0) var bloom: texture_2d<f32>;
@group(0) @binding(0) var bloom_sampler: sampler;

fn glowing(color: vec3<f32>, bloom: texture_2d<f32>, linear: sampler, uv: vec2<f32>,
        applied: vec4<f32>) -> vec3<f32> {
    if applied.y == 0.0 {
        return color;
    }
    return color
        + max(textureSampleLevel(bloom, linear, uv, 0.0).rgb * applied.x, vec3<f32>(0.0));
}

fn apply(color: vec3<f32>, at: PostPixel) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(bloom));
    let drawn = drawn_texels(size, post);
    return glowing(color, bloom, bloom_sampler, clamp(at.uv * drawn, vec2<f32>(0.5), drawn - 0.5)
        / size, piece().applied);
}
