import package::engine::post::frame::post;
import package::engine::post::pixel::{PostPixel, SceneLight};
import package::generated::applied::applied_light as scene_light;

@group(1) @binding(0) var scene_color: texture_2d<f32>;

@fragment
fn fs_lit(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let texel = textureLoad(scene_color, pixel, 0);
    let at = PostPixel(position.xy, position.xy / post.size.xy, pixel);
    let light = scene_light(SceneLight(texel.rgb, texel.a), at);
    return vec4<f32>(light.color, light.ambient);
}
