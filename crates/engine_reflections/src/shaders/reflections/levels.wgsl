import package::engine::post::frame::{drawn_uv, linear, post};
import package::engine::post::levels::halved;

@group(1) @binding(0) var source: texture_2d<f32>;

@vertex
fn vs_levels(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_levels_first(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let center = position.xy * 2.0;
    var colors: array<vec3<f32>, 5>;
    for (var i = 0u; i < 5u; i++) {
        var offset = vec2<f32>(0.0);
        if i > 0u {
            offset = vec2<f32>(f32(i & 1u) * 2.0 - 1.0, f32((i >> 1u) & 1u) * 2.0 - 1.0);
        }
        let at = drawn_uv(center + offset, post);
        let color = textureSampleLevel(source, linear, at, 0.0).rgb;
        colors[i] = color;
    }
    return vec4<f32>(
        (colors[0] * 4.0 + colors[4] + colors[3] + colors[1] + colors[2]) * 0.125, 1.0);
}

@fragment
fn fs_levels_down(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return halved(source, position.xy);
}
