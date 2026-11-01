import package::engine::color::{decode_srgb, dithered, encode_srgb};
import package::engine::post::pixel::{PostPixel, SceneLight};
import package::face::dither::dithering;
import package::generated::pieces::taa::piece;

fn turn() -> f32 {
    if dithering() {
        return piece().options.x;
    }
    return 0.0;
}

fn history(light: SceneLight, at: PostPixel) -> vec4<f32> {
    var kept = vec4<f32>(light.color, 1.0);
    if dithering() {
        kept = vec4<f32>(decode_srgb(dithered(encode_srgb(light.color),
            at.position, turn(), 1u)), 1.0);
    }
    return kept;
}
