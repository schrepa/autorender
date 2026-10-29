import package::frame::view::PI;

fn distribution(n_h: f32, alpha: f32) -> f32 {
    let alpha2 = alpha * alpha;
    let d = n_h * n_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * d * d);
}

fn visibility(n_v: f32, n_l: f32, alpha: f32) -> f32 {
    return 0.5 / mix(2.0 * n_l * n_v, n_l + n_v, alpha);
}

fn fresnel(f0: vec3<f32>, cosine: f32) -> vec3<f32> {
    let m = 1.0 - cosine;
    let m2 = m * m;
    return f0 + (vec3<f32>(1.0) - f0) * (m2 * m2 * m);
}

fn environment_brdf(f0: vec3<f32>, roughness: f32, n_v: f32) -> vec3<f32> {
    let r = roughness * vec4<f32>(-1.0, -0.0275, -0.572, 0.022)
        + vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let a004 = min(r.x * r.x, exp2(-9.28 * n_v)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}
