import package::engine::post::pixel::{PostPixel, SceneLight, ambient_blocked};

@group(0) @binding(0) var occlusion: texture_2d<f32>;

fn apply(light: SceneLight, at: PostPixel) -> SceneLight {
    return ambient_blocked(light, textureLoad(occlusion, at.pixel, 0).x);
}
