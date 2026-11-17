import package::engine::view::eye_nearer;

@group(0) @binding(0) var occlusion: texture_2d<f32>;

fn ambient_occlusion(pixel: vec2<f32>, world: vec3<f32>) -> f32 {
    let held = textureLoad(occlusion, vec2<i32>(pixel), 0).xy;
    if eye_nearer(world, held.y) {
        return 1.0;
    }
    return 1.0 - held.x;
}
