import package::engine::color::dithered;
import package::face::dither::dithering;
import package::post::pixel::PostPixel;

fn apply(color: vec3<f32>, at: PostPixel) -> vec3<f32> {
    if dithering() {
        return dithered(color, at.position, at.turn, at.salt);
    }
    return color;
}
