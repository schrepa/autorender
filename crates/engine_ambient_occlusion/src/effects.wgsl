import package::engine::post::frame::Post;

fn occlusion_at(occlusion: texture_2d<f32>, pixel: vec2<i32>, settings: Post) -> f32 {
    if settings.occlusion.w == 0.0 && settings.sky.w == 0.0 {
        return 1.0;
    }
    return textureLoad(occlusion, pixel, 0).x;
}

fn effect_shade(ambient: f32, visible: f32, settings: Post) -> f32 {
    if settings.occlusion.w == 0.0 && settings.sky.w == 0.0 {
        return 1.0;
    }
    return 1.0 - ambient * (1.0 - visible);
}
