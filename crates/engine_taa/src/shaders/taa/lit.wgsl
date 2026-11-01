import package::engine::post::frame::post;
import package::engine::post::pixel::{PostPixel, SceneLight};
import package::face::post_scene::apply as scene_light;
import package::generated::pieces::taa::piece;

@group(1) @binding(0) var scene_color: texture_2d<f32>;

fn turn() -> f32 {
    return piece().options.x;
}

fn taa_at(position: vec2<f32>, pixel: vec2<i32>) -> PostPixel {
    return PostPixel(position, position / post.size.xy, pixel, turn(), 2u);
}

@fragment
fn fs_lit(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let texel = textureLoad(scene_color, pixel, 0);
    let light = scene_light(SceneLight(texel.rgb, texel.a), taa_at(position.xy, pixel));
    return vec4<f32>(light.color, light.ambient);
}
