import package::engine::color::{filmic, luminance};
import package::face::sky::adapted_exposure;
import package::generated::pieces::pieces;
import package::post::pixel::PostPixel;

fn tonemap(color: vec3<f32>) -> vec3<f32> {
    let exposed = max(color, vec3<f32>(0.0)) * exposure();
    let peak = max(exposed.r, max(exposed.g, exposed.b));
    let kept = exposed * (filmic(vec3<f32>(peak)).r / max(peak, 1.0e-6));
    let curved = mix(filmic(exposed), kept, 0.5);
    let root = sqrt(curved);
    let perceptual = root + pieces.display.grade.x * (root * root * (3.0 - 2.0 * root) - root);
    var mapped = perceptual * perceptual;
    let tone = smoothstep(0.2, 0.8, sqrt(luminance(mapped)));
    mapped *= mix(pieces.display.shadow_tint.rgb, pieces.display.highlight_tint.rgb, tone);
    let top = max(mapped.r, max(mapped.g, mapped.b));
    let bottom = min(mapped.r, min(mapped.g, mapped.b));
    let muted = 1.0 - (top - bottom) / max(top, 1.0e-6);
    let amount = pieces.display.grade.z + pieces.display.grade.y * muted;
    return saturate(mix(vec3<f32>(luminance(mapped)), mapped, amount));
}

fn apply(color: vec3<f32>, at: PostPixel) -> vec3<f32> {
    return tonemap(color);
}

fn exposure() -> f32 {
    return select(adapted_exposure(pieces.display.grade.w), pieces.display.exposure.x,
        pieces.display.exposure.y != 0.0);
}
