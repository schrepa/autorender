import package::engine::color::luminance;
import package::engine::post::frame::{drawn_uv, post};
import package::engine::post::pixel::{PostPixel, SceneLight};
import package::generated::pieces::reflections::piece;

@group(0) @binding(0) var reflections: texture_2d<f32>;
@group(0) @binding(0) var reflections_sampler: sampler;

fn apply(light: SceneLight, at: PostPixel) -> SceneLight {
    if piece().applied.z == 0.0 {
        return light;
    }
    let reflection = textureSampleLevel(reflections, reflections_sampler,
        drawn_uv(at.uv * post.size.xy, post), 0.0);
    let strength = piece().applied.y;
    let replaced = saturate(reflection.a * strength);
    let color = light.color * (1.0 - light.ambient * replaced) + reflection.rgb * strength;
    let ambient = light.ambient * (1.0 - replaced) * luminance(light.color)
        / max(luminance(color), 1.0e-6);
    return SceneLight(color, ambient);
}
