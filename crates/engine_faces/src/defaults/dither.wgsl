import package::engine::color::encode_srgb;

fn dithering() -> bool {
    return false;
}

fn encoded(color: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    return encode_srgb(color);
}

fn stored(color: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    return color;
}
