import package::engine::color::encode_srgb;
import package::fluid::common::FrameUniform;

@group(1) @binding(0) var<uniform> post_frame: FrameUniform;
@group(1) @binding(1) var post_hdr: texture_2d<f32>;
@group(1) @binding(2) var post_bloom: texture_2d<f32>;
@group(1) @binding(3) var post_sampler: sampler;

@fragment
fn fs_bloom_extract(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(post_hdr));
    let uv = (pos.xy * 4.0) / dims;
    let avg = textureSampleLevel(post_hdr, post_sampler, uv, 0.0).rgb;

    let lum = max(avg.r, max(avg.g, avg.b));
    let thresh = 1.35;
    let knee = 0.45;
    let soft = clamp(lum - thresh + knee, 0.0, 2.0 * knee);
    let contrib = max(lum - thresh, (soft * soft) / (4.0 * knee + 1.0e-5)) / max(lum, 1.0e-5);
    return vec4<f32>(avg * contrib, 1.0);
}

fn bloom_blur(pos: vec2<f32>, dir: vec2<f32>) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(post_bloom));
    let uv = pos / dims;
    let step = dir / dims;
    var acc = textureSampleLevel(post_bloom, post_sampler, uv, 0.0).rgb * 0.227;
    let o1 = step * 1.42;
    let o2 = step * 3.35;
    let o3 = step * 5.45;
    acc += (
        textureSampleLevel(post_bloom, post_sampler, uv + o1, 0.0).rgb
        + textureSampleLevel(post_bloom, post_sampler, uv - o1, 0.0).rgb
    ) * 0.292;
    acc += (
        textureSampleLevel(post_bloom, post_sampler, uv + o2, 0.0).rgb
        + textureSampleLevel(post_bloom, post_sampler, uv - o2, 0.0).rgb
    ) * 0.078;
    acc += (
        textureSampleLevel(post_bloom, post_sampler, uv + o3, 0.0).rgb
        + textureSampleLevel(post_bloom, post_sampler, uv - o3, 0.0).rgb
    ) * 0.0165;
    return vec4<f32>(acc, 1.0);
}

@fragment
fn fs_bloom_h(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return bloom_blur(pos.xy, vec2<f32>(1.0, 0.0));
}

@fragment
fn fs_bloom_v(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return bloom_blur(pos.xy, vec2<f32>(0.0, 1.0));
}

fn agx_curve(v: vec3<f32>) -> vec3<f32> {
    return (((((15.5 * v - 40.14) * v + 31.96) * v - 6.868) * v + 0.4298) * v + 0.1191) * v - 0.00232;
}

fn agx_tonemap(color: vec3<f32>) -> vec3<f32> {
    let agx_in = mat3x3<f32>(
        vec3<f32>(0.84247906, 0.04232824, 0.04237565),
        vec3<f32>(0.07843360, 0.87846864, 0.07843360),
        vec3<f32>(0.07922375, 0.07916613, 0.87914297),
    );
    let agx_out = mat3x3<f32>(
        vec3<f32>(1.19687900, -0.05289685, -0.05297163),
        vec3<f32>(-0.09802088, 1.15190313, -0.09804345),
        vec3<f32>(-0.09902974, -0.09896118, 1.15107367),
    );
    var v = agx_in * max(color, vec3<f32>(1.0e-6));
    v = clamp(log2(v), vec3<f32>(-12.47393), vec3<f32>(4.026069));
    v = (v + 12.47393) * 0.060606064;
    v = agx_curve(v);
    v = agx_out * v;
    let luma = dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
    v = mix(vec3<f32>(luma), v, 1.18);
    return saturate(pow(max(v, vec3<f32>(0.0)), vec3<f32>(2.15)));
}

@fragment
fn fs_tonemap(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(post_hdr));
    let coord = clamp(vec2<i32>(pos.xy), vec2<i32>(0), dims - 1);
    let uv = (vec2<f32>(coord) + 0.5) / vec2<f32>(dims);

    let hdr = textureLoad(post_hdr, coord, 0).rgb;
    let bloom = textureSampleLevel(post_bloom, post_sampler, uv, 0.0).rgb;
    let mat_id = post_frame.style.x;
    var bloom_gain = 0.11;
    if mat_id > 0.5 && mat_id < 1.5 {
        bloom_gain = 0.14;
    } else if mat_id > 1.5 && mat_id < 2.5 {
        bloom_gain = 0.24;
    } else if mat_id > 2.5 {
        bloom_gain = 0.30;
    }
    var color = (hdr + bloom * bloom_gain) * 1.05;

    let centered = uv * 2.0 - 1.0;
    let vig = 1.0 - dot(centered, centered) * 0.14;
    color *= max(vig, 0.72);

    var srgb = encode_srgb(agx_tonemap(color));
    let dither = fract(52.9829189 * fract(dot(pos.xy, vec2<f32>(0.06711056, 0.00583715)))) - 0.5;
    srgb = saturate(srgb + dither * (1.0 / 255.0));
    return vec4<f32>(srgb, 1.0);
}
