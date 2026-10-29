import package::engine::color::dithered;
import package::post::frame::post;
import package::post::pixel::PostPixel;

fn apply(color: vec3<f32>, at: PostPixel) -> vec3<f32> {
    if post.sun_direction.w != 0.0 {
        return dithered(color, at.position, at.turn, at.salt);
    }
    return color;
}
