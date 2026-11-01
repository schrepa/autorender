import package::engine::post::pixel::{PostPixel, SceneLight};

fn history(light: SceneLight, at: PostPixel) -> vec4<f32> {
    return vec4<f32>(light.color, 1.0);
}
