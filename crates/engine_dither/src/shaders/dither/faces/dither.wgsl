import package::engine::color::{decode_srgb, dithered, encode_srgb};
import package::engine::view::camera;

fn dithering() -> bool {
    return true;
}

fn stored(color: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    return decode_srgb(dithered(encode_srgb(color), pixel, camera.options.z, 0u));
}
