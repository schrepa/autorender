import package::engine::color::{decode_srgb, dithered, encode_srgb, filmic, luminance};
import package::frame::view::{camera, lighting};

fn dithering() -> bool {
    return lighting.grade.w != 0.0;
}

fn encoded(color: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    let value = encode_srgb(color);
    if !dithering() {
        return value;
    }
    return dithered(value, pixel, camera.options.z, 0u);
}

fn stored(color: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    if !dithering() {
        return color;
    }
    return decode_srgb(dithered(encode_srgb(color), pixel, camera.options.z, 0u));
}

fn grade(color: vec3<f32>) -> vec3<f32> {
    let exposed = max(color, vec3<f32>(0.0)) * lighting.horizon.w;
    let peak = max(exposed.r, max(exposed.g, exposed.b));
    let kept = exposed * (filmic(vec3<f32>(peak)).r / max(peak, 1.0e-6));
    let curved = mix(filmic(exposed), kept, 0.5);
    let root = sqrt(curved);
    let perceptual = root + lighting.grade.x * (root * root * (3.0 - 2.0 * root) - root);
    var mapped = perceptual * perceptual;
    let tone = smoothstep(0.2, 0.8, sqrt(luminance(mapped)));
    mapped *= mix(lighting.shadow_tint.rgb, lighting.highlight_tint.rgb, tone);
    let top = max(mapped.r, max(mapped.g, mapped.b));
    let bottom = min(mapped.r, min(mapped.g, mapped.b));
    let muted = 1.0 - (top - bottom) / max(top, 1.0e-6);
    let amount = lighting.ground.w + lighting.grade.y * muted;
    return saturate(mix(vec3<f32>(luminance(mapped)), mapped, amount));
}

fn glow_weight(color: vec3<f32>) -> f32 {
    let bright = max(luminance(color) * lighting.horizon.w - lighting.effects.y, 0.0);
    return bright / (1.0 + bright);
}
