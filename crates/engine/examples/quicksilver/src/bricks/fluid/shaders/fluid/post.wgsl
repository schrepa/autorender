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
    let dx = 1.5 / dims.x;
    let dy = 1.5 / dims.y;
    let c0 = textureSampleLevel(post_hdr, post_sampler, uv + vec2<f32>(-dx, -dy), 0.0).rgb;
    let c1 = textureSampleLevel(post_hdr, post_sampler, uv + vec2<f32>(dx, -dy), 0.0).rgb;
    let c2 = textureSampleLevel(post_hdr, post_sampler, uv + vec2<f32>(-dx, dy), 0.0).rgb;
    let c3 = textureSampleLevel(post_hdr, post_sampler, uv + vec2<f32>(dx, dy), 0.0).rgb;
    let avg = (c0 + c1 + c2 + c3) * 0.25;

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
    let offsets = array<f32, 5>(0.0, 1.3846154, 3.2307692, 5.15, 7.1);
    let weights = array<f32, 5>(0.215, 0.295, 0.180, 0.085, 0.0325);
    var acc = textureSampleLevel(post_bloom, post_sampler, uv, 0.0).rgb * weights[0];
    for (var i = 1; i < 5; i += 1) {
        let o = step * offsets[i];
        acc += textureSampleLevel(post_bloom, post_sampler, uv + o, 0.0).rgb * weights[i];
        acc += textureSampleLevel(post_bloom, post_sampler, uv - o, 0.0).rgb * weights[i];
    }
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
    let v2 = v * v;
    let v4 = v2 * v2;
    return 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v - 0.00232;
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
    v = (v - (-12.47393)) / (4.026069 - (-12.47393));
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
    let dither = fract(sin(dot(pos.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453) - 0.5;
    srgb = saturate(srgb + dither * (1.0 / 255.0));
    return vec4<f32>(srgb, 1.0);
}
