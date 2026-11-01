import package::engine::post::pixel::PostPixel;
import package::sky::piece::sky_piece;

@group(0) @binding(0) var haze: texture_2d<f32>;

fn haze_at(hazed: texture_2d<f32>, pixel: vec2<i32>) -> vec3<f32> {
    if sky_piece.haze.w == 0.0 {
        return vec3<f32>(0.0);
    }
    return textureLoad(hazed, pixel, 0).rgb;
}

fn with_haze(color: vec3<f32>, light: vec3<f32>) -> vec3<f32> {
    if sky_piece.haze.w == 0.0 {
        return color;
    }
    return color + light;
}

fn apply(color: vec3<f32>, at: PostPixel) -> vec3<f32> {
    return with_haze(color, haze_at(haze, at.pixel));
}
