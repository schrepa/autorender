import package::engine::view::eye_nearer;
import package::generated::pieces::shadows::piece;

@group(0) @binding(0) var sky_open: texture_2d<f32>;

fn sky_visibility(pixel: vec2<f32>, world: vec3<f32>) -> f32 {
    if piece().sky.w == 0.0 {
        return 1.0;
    }
    let held = textureLoad(sky_open, vec2<i32>(pixel), 0).xy;
    if eye_nearer(world, held.y) {
        return 1.0;
    }
    return 1.0 - held.x;
}
