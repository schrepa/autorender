import package::engine::color::{decode_srgb, dithered, encode_srgb};
import package::engine::post::pixel::PostPixel;

fn apply(color: vec3<f32>, at: PostPixel) -> vec3<f32> {
    return decode_srgb(dithered(encode_srgb(color), at.position, at.turn, at.salt));
}
