import package::engine::color::{decode_srgb, dithered, encode_srgb};
import package::frame::view::camera;
import package::generated::pieces::pieces;

fn dithering() -> bool {
    return pieces.dither.flags.x != 0.0;
}

fn stored(color: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    if !dithering() {
        return color;
    }
    return decode_srgb(dithered(encode_srgb(color), pixel, camera.options.z, 0u));
}
