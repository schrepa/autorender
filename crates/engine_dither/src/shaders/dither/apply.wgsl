import package::engine::color::{decode_srgb, encode_srgb, hash3};
import package::engine::post::pixel::PostPixel;
import package::engine::view::noise_phase;

fn dithered(encoded: vec3<f32>, pixel: vec2<f32>, turn: f32) -> vec3<f32> {
    let bits = hash3(vec3<u32>(vec2<u32>(max(pixel, vec2<f32>(0.0))), bitcast<u32>(turn)));
    let first = vec3<f32>(bits >> vec3<u32>(16u)) * (1.0 / 65536.0) - 0.5;
    let second = vec3<f32>(bits & vec3<u32>(0xffffu)) * (1.0 / 65536.0) - 0.5;
    let room = saturate(min(encoded, vec3<f32>(1.0) - encoded) * 510.0);
    return encoded + (first + second * room) * (1.0 / 255.0);
}

fn apply(color: vec3<f32>, at: PostPixel) -> vec3<f32> {
    return decode_srgb(dithered(encode_srgb(color), at.position, noise_phase()));
}
