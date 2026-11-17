import package::engine::post::frame::{depth_texture, drawn_texels, drawn_uv, linear, post, scene};

@group(1) @binding(0) var source: texture_2d<f32>;

@fragment
fn fs_levels_scene(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    if !(textureLoad(depth_texture, pixel, 0) > 0.0) {
        discard;
        return vec4<f32>(0.0);
    }
    return vec4<f32>(textureLoad(scene, pixel, 0).rgb, 1.0);
}

@fragment
fn fs_levels_first(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let center = position.xy * 2.0;
    return (first_tap(center, vec2<f32>(0.0)) * 4.0 + first_tap(center, vec2<f32>(-1.0, -1.0))
        + first_tap(center, vec2<f32>(1.0, 1.0)) + first_tap(center, vec2<f32>(1.0, -1.0))
        + first_tap(center, vec2<f32>(-1.0, 1.0))) * 0.125;
}

fn first_tap(center: vec2<f32>, offset: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(source, linear, drawn_uv(center + offset, post), 0.0);
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
