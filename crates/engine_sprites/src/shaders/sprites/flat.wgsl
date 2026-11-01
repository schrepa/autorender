import package::engine::shading::{Shaded, ViewPixel};
import package::generated::targets::{ViewTargets, view_targets};

@group(1) @binding(0) var checker: texture_2d<f32>;
@group(1) @binding(1) var checker_sampler: sampler;

struct SpriteOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) ndc: vec2<f32>,
}

@vertex
fn vs_sprite(@builtin(vertex_index) vertex: u32) -> SpriteOutput {
    let corner = vec2<f32>(f32((50u >> vertex) & 1u), f32((44u >> vertex) & 1u));
    var output: SpriteOutput;
    output.ndc = corner - vec2<f32>(0.5);
    output.clip_position = vec4<f32>(output.ndc, 0.5, 1.0);
    output.uv = vec2<f32>(corner.x, 1.0 - corner.y);
    return output;
}

@fragment
fn fs_sprite(input: SpriteOutput) -> ViewTargets {
    let texel = textureSample(checker, checker_sampler, input.uv);
    var shaded: Shaded;
    shaded.linear = texel.rgb;
    shaded.normal = vec3<f32>(0.0, 0.0, 1.0);
    let at = vec4<f32>(input.ndc, 0.0, 1.0);
    return view_targets(shaded, ViewPixel(at, at));
}
