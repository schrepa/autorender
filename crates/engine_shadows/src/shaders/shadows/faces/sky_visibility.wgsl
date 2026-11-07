import package::generated::pieces::shadows::piece;

@group(0) @binding(0) var sky_open: texture_2d<f32>;

fn sky_visibility(pixel: vec2<f32>) -> f32 {
    if piece().sky.w == 0.0 {
        return 1.0;
    }
    return 1.0 - textureLoad(sky_open, vec2<i32>(pixel), 0).x;
}
