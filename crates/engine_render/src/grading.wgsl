// The existing display grade, applied once to exposed, composed radiance.
// Parameters are contrast weight, saturation and vibrance; tints have unit
// luminance. Exposure belongs to the radiance producers, not this function.
fn filmic(x: vec3<f32>) -> vec3<f32> {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

fn grade_exposed(exposed: vec3<f32>, parameters: vec3<f32>, shadows: vec3<f32>,
        highlights: vec3<f32>) -> vec3<f32> {
    let peak = max(exposed.r, max(exposed.g, exposed.b));
    let kept = exposed * (filmic(vec3<f32>(peak)).r / max(peak, 1.0e-6));
    let curved = mix(filmic(exposed), kept, 0.5);
    let root = sqrt(curved);
    let perceptual = root + parameters.x * (root * root * (3.0 - 2.0 * root) - root);
    var mapped = perceptual * perceptual;
    let tone = smoothstep(0.2, 0.8, sqrt(luminance(mapped)));
    mapped *= mix(shadows, highlights, tone);
    let top = max(mapped.r, max(mapped.g, mapped.b));
    let bottom = min(mapped.r, min(mapped.g, mapped.b));
    let muted = 1.0 - (top - bottom) / max(top, 1.0e-6);
    let amount = parameters.y + parameters.z * muted;
    return saturate(mix(vec3<f32>(luminance(mapped)), mapped, amount));
}
