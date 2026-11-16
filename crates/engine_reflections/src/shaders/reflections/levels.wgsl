import package::engine::post::frame::{depth_texture, drawn_texels, drawn_uv, linear, post, scene};

@group(1) @binding(0) var source: texture_2d<f32>;

@fragment
fn fs_levels_scene(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let solid = select(0.0, 1.0, textureLoad(depth_texture, pixel, 0) > 0.0);
    return vec4<f32>(textureLoad(scene, pixel, 0).rgb * solid, solid);
}

@fragment
fn fs_levels_first(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let center = position.xy * 2.0;
    var colors: array<vec4<f32>, 5>;
    for (var i = 0u; i < 5u; i++) {
        var offset = vec2<f32>(0.0);
        if i > 0u {
            offset = vec2<f32>(f32(i & 1u) * 2.0 - 1.0, f32((i >> 1u) & 1u) * 2.0 - 1.0);
        }
        let at = drawn_uv(center + offset, post);
        colors[i] = textureSampleLevel(source, linear, at, 0.0);
    }
    return (colors[0] * 4.0 + colors[4] + colors[3] + colors[1] + colors[2]) * 0.125;
}

fn level_tap(at: vec2<f32>, drawn: vec2<f32>, size: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(source, linear, clamp(at, vec2<f32>(0.5), drawn - 0.5) / size, 0.0);
}

@fragment
fn fs_levels_down(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(source));
    let drawn = drawn_texels(size, post);
    let at = position.xy * 2.0;
    let sum = level_tap(at, drawn, size) * 4.0
        + level_tap(at - 1.0, drawn, size)
        + level_tap(at + 1.0, drawn, size)
        + level_tap(at + vec2<f32>(1.0, -1.0), drawn, size)
        + level_tap(at + vec2<f32>(-1.0, 1.0), drawn, size);
    return sum * 0.125;
}
