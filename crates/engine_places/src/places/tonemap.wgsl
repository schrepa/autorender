import package::engine::post::pixel::PostPixel;

fn tonemap(color: vec3<f32>) -> vec3<f32> {
    return color;
}

fn apply(color: vec3<f32>, at: PostPixel) -> vec3<f32> {
    return color;
}

fn exposure() -> f32 {
    return 1.0;
}

fn unfilmic(shown: f32) -> f32 {
    return shown;
}
