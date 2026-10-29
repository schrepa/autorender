import package::post::frame::{Post, post};
import package::post::pixel::PostPixel;

@group(0) @binding(19) var haze: texture_2d<f32>;

fn haze_at(hazed: texture_2d<f32>, pixel: vec2<i32>, settings: Post) -> vec3<f32> {
    if settings.haze.w == 0.0 {
        return vec3<f32>(0.0);
    }
    return textureLoad(hazed, pixel, 0).rgb;
}

fn with_haze(color: vec3<f32>, light: vec3<f32>, settings: Post) -> vec3<f32> {
    if settings.haze.w == 0.0 {
        return color;
    }
    let share = light / (vec3<f32>(1.0) + abs(light));
    let added = vec3<f32>(1.0) - (vec3<f32>(1.0) - color) * (vec3<f32>(1.0) - share);
    return select(color * (vec3<f32>(1.0) + share), added, share >= vec3<f32>(0.0));
}

fn apply(color: vec3<f32>, at: PostPixel) -> vec3<f32> {
    return with_haze(color, haze_at(haze, at.pixel, post), post);
}
