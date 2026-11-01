import package::engine::color::{decode_srgb, dithered, encode_srgb};
import package::engine::post::pixel::{PostPixel, SceneLight};
import package::face::dither::dithering;

fn history(light: SceneLight, at: PostPixel) -> vec4<f32> {
    var kept = vec4<f32>(light.color, 1.0);
    if dithering() {
        kept = vec4<f32>(decode_srgb(dithered(encode_srgb(light.color),
            at.position, at.turn, 1u)), 1.0);
    }
    return kept;
}
