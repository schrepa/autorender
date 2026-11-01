import package::engine::color::{filmic, luminance};
import package::engine::post::pixel::PostPixel;
import package::face::exposure::adapted;
import package::generated::pieces::tonemap::piece;

fn tonemap(color: vec3<f32>) -> vec3<f32> {
    let exposed = max(color, vec3<f32>(0.0)) * exposure();
    let peak = max(exposed.r, max(exposed.g, exposed.b));
    let kept = exposed * (filmic(vec3<f32>(peak)).r / max(peak, 1.0e-6));
    let curved = mix(filmic(exposed), kept, 0.5);
    let root = sqrt(curved);
    let perceptual = root + piece().grade.x * (root * root * (3.0 - 2.0 * root) - root);
    var mapped = perceptual * perceptual;
    let tone = smoothstep(0.2, 0.8, sqrt(luminance(mapped)));
    mapped *= mix(piece().shadow_tint.rgb, piece().highlight_tint.rgb, tone);
    let top = max(mapped.r, max(mapped.g, mapped.b));
    let bottom = min(mapped.r, min(mapped.g, mapped.b));
    let muted = 1.0 - (top - bottom) / max(top, 1.0e-6);
    let amount = piece().grade.z + piece().grade.y * muted;
    return saturate(mix(vec3<f32>(luminance(mapped)), mapped, amount));
}

fn apply(color: vec3<f32>, at: PostPixel) -> vec3<f32> {
    return tonemap(color);
}

fn exposure() -> f32 {
    return adapted(piece().exposure.x);
}

fn unfilmic(shown: f32) -> f32 {
    let a = 2.51 - 2.43 * shown;
    let b = 0.03 - 0.59 * shown;
    return (sqrt(b * b + 0.56 * a * shown) - b) / (2.0 * a);
}
