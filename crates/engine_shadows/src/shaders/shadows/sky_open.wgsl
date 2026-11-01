import package::engine::post::pixel::{PostPixel, SceneLight, ambient_blocked};
import package::generated::pieces::shadows::piece;

@group(0) @binding(0) var sky_open: texture_2d<f32>;

fn apply(light: SceneLight, at: PostPixel) -> SceneLight {
    if piece().sky.w == 0.0 {
        return light;
    }
    return ambient_blocked(light, textureLoad(sky_open, at.pixel, 0).x);
}
