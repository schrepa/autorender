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
    return SceneLight(min(light.color * (1.0 - saturate(reflection.a * strength))
        + reflection.rgb * strength, vec3<f32>(1.0)), light.ambient);
}
