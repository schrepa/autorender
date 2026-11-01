import package::engine::post::pixel::{PostPixel, SceneLight};

@group(0) @binding(0) var occlusion: texture_2d<f32>;

fn effect_shade(ambient: f32, visible: f32) -> f32 {
    return 1.0 - ambient * (1.0 - visible);
}

fn apply(light: SceneLight, at: PostPixel) -> SceneLight {
    let visible = textureLoad(occlusion, at.pixel, 0).x;
    return SceneLight(light.color * effect_shade(light.ambient, visible), light.ambient);
}
