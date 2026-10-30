import package::engine::color::decode_srgb;
import package::post::pixel::PostPixel;

@group(0) @binding(0) var accumulated: texture_2d<f32>;
@group(0) @binding(1) var revealed: texture_2d<f32>;

struct Weighted {
    color: vec3<f32>,
    reveal: f32,
    at: PostPixel,
}

fn weighted(position: vec4<f32>) -> Weighted {
    let pixel = vec2<i32>(position.xy);
    let reveal = textureLoad(revealed, pixel, 0).r;
    let sum = textureLoad(accumulated, pixel, 0);
    let color = min(sum.rgb / max(sum.a, 1.0e-5), vec3<f32>(1.0));
    let uv = position.xy / vec2<f32>(textureDimensions(revealed));
    return Weighted(color, reveal, PostPixel(position.xy, uv, pixel, 0.0, 3u));
}

fn output(drawn: Weighted, encoded: vec3<f32>, shown: vec3<f32>) -> vec4<f32> {
    var color = drawn.color;
    if any(shown != encoded) {
        color = decode_srgb(shown);
    }
    return vec4<f32>(color, 1.0 - drawn.reveal);
}
