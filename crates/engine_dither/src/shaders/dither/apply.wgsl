import package::engine::color::dithered;
import package::engine::post::pixel::PostPixel;

fn apply(color: vec3<f32>, at: PostPixel) -> vec3<f32> {
    return dithered(color, at.position, at.turn, at.salt);
}
