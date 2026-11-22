const POST_FLAG_ACES: u32 = 1u;
const POST_FLAG_LINEAR_OUT: u32 = 2u;
const POST_FLAG_DISABLE_BLOOM: u32 = 4u;
const POST_FLAG_AGX_PUNCH: u32 = 8u;
const POST_FLAG_DISABLE_SHAFTS: u32 = 16u;
const POST_FLAG_DISABLE_DITHER: u32 = 32u;
const POST_FLAG_STANDARD_Z: u32 = 64u;

fn post_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn safe_pow3(v: vec3<f32>, exponent: f32) -> vec3<f32> {
    let positive = max(v, vec3<f32>(1.0e-7));
    return select(
        pow(positive, vec3<f32>(exponent)),
        vec3<f32>(0.0),
        v <= vec3<f32>(0.0),
    );
}

fn agx_default_contrast_approx(x: vec3<f32>) -> vec3<f32> {
    return (((((15.5 * x - 40.14) * x + 31.96) * x - 6.868) * x + 0.4298) * x + 0.1191) * x
        - 0.00232;
}

fn agx_encode(hdr: vec3<f32>) -> vec3<f32> {
    let agx_in = mat3x3<f32>(
        vec3<f32>(0.84247906, 0.04232824, 0.04237565),
        vec3<f32>(0.07843360, 0.87846864, 0.07843360),
        vec3<f32>(0.07922375, 0.07916613, 0.87914297),
    );
    let agx_out = mat3x3<f32>(
        vec3<f32>(1.19687900, -0.05289685, -0.05297163),
        vec3<f32>(-0.09802088, 1.15190313, -0.09804345),
        vec3<f32>(-0.09902975, -0.09896118, 1.15107367),
    );
    var v = max(agx_in * max(hdr, vec3<f32>(0.0)), vec3<f32>(1.0e-6));
    v = clamp((log2(v) + 12.47393) * 0.060606064, vec3<f32>(0.0), vec3<f32>(1.0));
    v = agx_default_contrast_approx(v);
    return max(agx_out * v, vec3<f32>(0.0));
}

fn agx_tonemap(hdr: vec3<f32>) -> vec3<f32> {
    return saturate(safe_pow3(agx_encode(hdr), 2.2));
}

fn agx_tonemap_look(
    hdr: vec3<f32>,
    exposure: f32,
    saturation: f32,
    punch: bool,
) -> vec3<f32> {
    let exposed = max(hdr, vec3<f32>(0.0)) * max(exposure, 0.0);
    var mapped: vec3<f32>;
    if punch {
        let v = safe_pow3(agx_encode(exposed), 1.18);
        let luma_d = post_luma(v);
        let punched = max(mix(vec3<f32>(luma_d), v, 1.2), vec3<f32>(0.0));
        mapped = saturate(safe_pow3(punched, 2.2));
    } else {
        mapped = agx_tonemap(exposed);
    }
    let luma = post_luma(mapped);
    return saturate(mix(vec3<f32>(luma), mapped, saturation));
}

fn aces_tonemap(hdr: vec3<f32>) -> vec3<f32> {
    let aces_in = mat3x3<f32>(
        vec3<f32>(0.59719, 0.07600, 0.02840),
        vec3<f32>(0.35458, 0.90834, 0.13383),
        vec3<f32>(0.04823, 0.01566, 0.83777),
    );
    let aces_out = mat3x3<f32>(
        vec3<f32>(1.60475, -0.10208, -0.00327),
        vec3<f32>(-0.53108, 1.10813, -0.07276),
        vec3<f32>(-0.07367, -0.00605, 1.07602),
    );
    let v = aces_in * max(hdr, vec3<f32>(0.0));
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return saturate(aces_out * (a / b));
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let clamped = max(color, vec3<f32>(0.0));
    let low = clamped * 12.92;
    let high = 1.055 * pow(max(clamped, vec3<f32>(1.0e-7)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return saturate(select(high, low, clamped <= vec3<f32>(0.0031308)));
}

fn apply_vignette(color: vec3<f32>, uv: vec2<f32>, strength: f32) -> vec3<f32> {
    let centered = (uv * 2.0 - 1.0) * vec2<f32>(0.82, 0.94);
    let dist_sq = dot(centered, centered);
    let vig = clamp(1.0 - dist_sq * max(strength, 0.0), 0.35, 1.0);
    return color * vig;
}

fn interleaved_gradient_noise(pixel: vec2<f32>, time: f32) -> f32 {
    let frame = fract(time * 60.0) * 64.0;
    let p = pixel + vec2<f32>(frame * 5.588238, frame * 5.588238);
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

fn apply_film_grain(
    color: vec3<f32>,
    pixel: vec2<f32>,
    time: f32,
    intensity: f32,
) -> vec3<f32> {
    let n0 = interleaved_gradient_noise(pixel, time);
    let n1 = interleaved_gradient_noise(pixel + vec2<f32>(17.0, 31.0), time + 0.37);
    let tri = n0 + n1 - 1.0;
    let luma = post_luma(color);
    let midtone_mask = 1.0 - abs(luma * 2.0 - 1.0) * 0.5;
    return saturate(color + vec3<f32>(tri * intensity * midtone_mask));
}
