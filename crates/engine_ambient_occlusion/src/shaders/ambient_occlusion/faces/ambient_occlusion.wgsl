@group(0) @binding(0) var occlusion: texture_2d<f32>;

fn ambient_occlusion(pixel: vec2<f32>) -> f32 {
    return 1.0 - textureLoad(occlusion, vec2<i32>(pixel), 0).x;
}
