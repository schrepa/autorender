@group(0) @binding(0) var accumulated: texture_2d<f32>;
@group(0) @binding(1) var revealed: texture_2d<f32>;

override TARGET_IS_SRGB: bool = false;
override DITHER: bool = false;

@fragment
fn fs_composite(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let reveal = textureLoad(revealed, pixel, 0).r;
    if reveal >= 1.0 {
        discard;
    }
    let sum = textureLoad(accumulated, pixel, 0);
    var color = min(sum.rgb / max(sum.a, 1.0e-5), vec3<f32>(1.0));
    if !TARGET_IS_SRGB {
        color = encode_srgb(color);
        if DITHER {
            color = dithered(color, position.xy, 0.0, 3u);
        }
    } else if DITHER {
        color = decode_srgb(dithered(encode_srgb(color), position.xy, 0.0, 3u));
    }
    return vec4<f32>(color, 1.0 - reveal);
}
