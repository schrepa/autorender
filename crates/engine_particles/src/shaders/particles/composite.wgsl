import package::engine::color::{decode_srgb, encode_srgb};
import package::engine::post::pixel::PostPixel;
import package::face::post_display::apply;

@group(1) @binding(0) var accumulated: texture_2d<f32>;
@group(1) @binding(1) var revealed: texture_2d<f32>;

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

@vertex
fn vs_composite(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_composite(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let drawn = weighted(position);
    if drawn.reveal >= 1.0 {
        discard;
    }
    let encoded = encode_srgb(drawn.color);
    var shown = encoded;
    shown = apply(shown, drawn.at);
    return output(drawn, encoded, shown);
}
