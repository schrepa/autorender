import package::engine::color::luminance;
import package::engine::post::pixel::PostPixel;
import package::face::sky::{ambient, sky_average};
import package::face::sun::sun_color;
import package::generated::pieces::tonemap::piece;
import package::tonemap::faces::tonemap::exposure;

fn curved(x: vec3<f32>) -> vec3<f32> {
    let curve = piece().curve;
    let within = clamp(x, vec3<f32>(1.0e-9), vec3<f32>(piece().exposure.z));
    return saturate(pow(within, vec3<f32>(curve.x))
        / (pow(within, vec3<f32>(curve.y)) * curve.z + curve.w));
}

fn balanced(color: vec3<f32>) -> vec3<f32> {
    let amount = piece().exposure.y;
    if amount <= 0.0 {
        return color;
    }
    let white = sun_color() + sky_average(vec3<f32>(0.0, 1.0, 0.0)) * ambient();
    if any(white <= vec3<f32>(0.0)) {
        return color;
    }
    return color * pow(vec3<f32>(luminance(white)) / white, vec3<f32>(amount));
}

fn tonemap(color: vec3<f32>) -> vec3<f32> {
    let exposed = balanced(max(color, vec3<f32>(0.0))) * exposure();
    let peak = max(exposed.r, max(exposed.g, exposed.b));
    let shown = curved(vec3<f32>(peak)).r;
    let faded = 1.0 - 1.0 / (piece().grade.z * max(peak - shown, 0.0) + 1.0);
    let kept = mix(exposed * (shown / max(peak, 1.0e-6)), vec3<f32>(shown), faded);
    var mapped = mix(kept, curved(exposed), piece().grade.w);
    let tone = smoothstep(piece().shadow_tint.w, piece().highlight_tint.w,
        sqrt(luminance(mapped)));
    mapped *= mix(piece().shadow_tint.rgb, piece().highlight_tint.rgb, tone);
    let top = max(mapped.r, max(mapped.g, mapped.b));
    let bottom = min(mapped.r, min(mapped.g, mapped.b));
    let muted = 1.0 - (top - bottom) / max(top, 1.0e-6);
    let amount = piece().grade.y + piece().grade.x * muted;
    return saturate(mix(vec3<f32>(luminance(mapped)), mapped, amount));
}

fn apply(color: vec3<f32>, at: PostPixel) -> vec3<f32> {
    return tonemap(color);
}
