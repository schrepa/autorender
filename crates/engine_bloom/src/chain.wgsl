import package::engine::color::luminance;
import package::post::frame::{drawn_texels, drawn_uv, glow, linear, post, scene};

@group(0) @binding(4) var source: texture_2d<f32>;

fn level_tap(level: texture_2d<f32>, at: vec2<f32>, drawn: vec2<f32>, size: vec2<f32>)
        -> vec3<f32> {
    return textureSampleLevel(level, linear, clamp(at, vec2<f32>(0.5), drawn - 0.5) / size,
        0.0).rgb;
}

fn glow_light(uv: vec2<f32>) -> vec3<f32> {
    let color = textureSampleLevel(scene, linear, uv, 0.0).rgb;
    let weight = min(textureSampleLevel(glow, linear, uv, 0.0).r, 0.98);
    return color * (weight / (1.0 - weight));
}

@fragment
fn fs_bloom_first(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let center = position.xy * 2.0;
    var sum = vec3<f32>(0.0);
    var weights = 0.0;
    for (var i = 0u; i < 5u; i++) {
        var offset = vec2<f32>(0.0);
        if i > 0u {
            offset = vec2<f32>(f32(i & 1u) * 2.0 - 1.0, f32((i >> 1u) & 1u) * 2.0 - 1.0);
        }
        let light = glow_light(drawn_uv(center + offset, post));
        let weight = select(1.0, 4.0, i == 0u) / (1.0 + luminance(light));
        sum += light * weight;
        weights += weight;
    }
    return vec4<f32>(sum / weights, 1.0);
}

fn halved(level: texture_2d<f32>, position: vec2<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(level));
    let drawn = drawn_texels(size, post);
    let at = position * 2.0;
    let sum = level_tap(level, at, drawn, size) * 4.0
        + level_tap(level, at - 1.0, drawn, size)
        + level_tap(level, at + 1.0, drawn, size)
        + level_tap(level, at + vec2<f32>(1.0, -1.0), drawn, size)
        + level_tap(level, at + vec2<f32>(-1.0, 1.0), drawn, size);
    return vec4<f32>(sum * 0.125, 1.0);
}

@fragment
fn fs_bloom_down(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return halved(source, position.xy);
}

@fragment
fn fs_bloom_up(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(source));
    let drawn = drawn_texels(size, post);
    let at = position.xy * 0.5;
    let sum = (level_tap(source, at + vec2<f32>(-2.0, 0.0), drawn, size)
        + level_tap(source, at + vec2<f32>(2.0, 0.0), drawn, size)
        + level_tap(source, at + vec2<f32>(0.0, -2.0), drawn, size)
        + level_tap(source, at + vec2<f32>(0.0, 2.0), drawn, size))
        + (level_tap(source, at - 1.0, drawn, size)
        + level_tap(source, at + 1.0, drawn, size)
        + level_tap(source, at + vec2<f32>(1.0, -1.0), drawn, size)
        + level_tap(source, at + vec2<f32>(-1.0, 1.0), drawn, size)) * 2.0;
    return vec4<f32>(sum / 12.0, 1.0);
}
