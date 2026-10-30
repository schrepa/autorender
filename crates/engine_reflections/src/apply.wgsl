import package::generated::pieces::pieces;
import package::post::frame::{drawn_uv, post};
import package::post::pixel::{PostPixel, SceneLight};

@group(0) @binding(8) var reflections: texture_2d<f32>;
@group(0) @binding(21) var reflections_sampler: sampler;

fn apply(light: SceneLight, at: PostPixel) -> SceneLight {
    if pieces.reflections.applied.z == 0.0 {
        return light;
    }
    let reflection = textureSampleLevel(reflections, reflections_sampler,
        drawn_uv(at.uv * post.size.xy, post), 0.0);
    let strength = pieces.reflections.applied.y;
    return SceneLight(min(light.color * (1.0 - saturate(reflection.a * strength))
        + reflection.rgb * strength, vec3<f32>(1.0)), light.ambient);
}
