import package::engine::color::encode_srgb;
import package::generated::frame::FrameUniforms;

@group(2) @binding(0) var scene_color: texture_2d<f32>;
@group(2) @binding(1) var scene_bloom: texture_2d<f32>;
@group(2) @binding(2) var tone_sampler: sampler;
@group(2) @binding(3) var<uniform> frame_params: FrameUniforms;

fn agx_default_contrast(x: vec3<f32>) -> vec3<f32> {
    return (((((15.5 * x - 40.14) * x + 31.96) * x - 6.868) * x + 0.4298) * x + 0.1191) * x - 0.00232;
}

fn agx_tonemap(val_in: vec3<f32>) -> vec3<f32> {
    let inset = mat3x3<f32>(
        vec3<f32>(0.84247906, 0.04232824, 0.04237565),
        vec3<f32>(0.07843360, 0.87846864, 0.07843360),
        vec3<f32>(0.07922375, 0.07916613, 0.87914297),
    );
    let outset = mat3x3<f32>(
        vec3<f32>(1.19687900, -0.05289685, -0.05297163),
        vec3<f32>(-0.09802088, 1.15190313, -0.09804345),
        vec3<f32>(-0.09902975, -0.09896118, 1.15107367),
    );
    var v = max(inset * max(val_in, vec3<f32>(0.0)), vec3<f32>(1.0e-6));
    v = clamp((log2(v) + 12.47393) * (1.0 / 16.5), vec3<f32>(0.0), vec3<f32>(1.0));
    v = agx_default_contrast(v);
    v = max(outset * v, vec3<f32>(0.0));
    return saturate(pow(v, vec3<f32>(2.2)));
}

fn rcas_luma(c: vec3<f32>) -> f32 {
    return c.g * 0.5 + (c.r + c.b) * 0.25;
}

fn sample_rcas_hdr(pixel: vec2<i32>, max_px: vec2<i32>) -> vec3<f32> {
    let e = max(textureLoad(scene_color, pixel, 0).rgb, vec3<f32>(0.0));
    let ee = rcas_luma(e);
    var mn = ee;
    var mx = ee;
    var neigh_sum = vec3<f32>(0.0);

    let t0 = max(textureLoad(scene_color, vec2<i32>(pixel.x, max(pixel.y - 1, 0)), 0).rgb, vec3<f32>(0.0));
    let l0 = rcas_luma(t0);
    mn = min(mn, l0);
    mx = max(mx, l0);
    neigh_sum += t0;

    let t1 = max(textureLoad(scene_color, vec2<i32>(max(pixel.x - 1, 0), pixel.y), 0).rgb, vec3<f32>(0.0));
    let l1 = rcas_luma(t1);
    mn = min(mn, l1);
    mx = max(mx, l1);
    neigh_sum += t1;

    let t2 = max(textureLoad(scene_color, vec2<i32>(min(pixel.x + 1, max_px.x), pixel.y), 0).rgb, vec3<f32>(0.0));
    let l2 = rcas_luma(t2);
    mn = min(mn, l2);
    mx = max(mx, l2);
    neigh_sum += t2;

    let t3 = max(textureLoad(scene_color, vec2<i32>(pixel.x, min(pixel.y + 1, max_px.y)), 0).rgb, vec3<f32>(0.0));
    let l3 = rcas_luma(t3);
    mn = min(mn, l3);
    mx = max(mx, l3);
    neigh_sum += t3;

    let amp = sqrt(clamp(mn / max(mx, 1.0e-4), 0.0, 1.0)) * (1.0 / (1.0 + mx * 0.35));
    let w = -amp * 0.14;
    let sharp = (e + neigh_sum * w) / max(1.0 + 4.0 * w, 0.1);
    return max(sharp, vec3<f32>(0.0));
}

@fragment
fn fs_tone(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(scene_color));
    let max_px = size - vec2<i32>(1);
    let pixel = clamp(vec2<i32>(pos.xy), vec2<i32>(0), max_px);

    if frame_params.fog_params.w > 0.5 {
        let raw = textureLoad(scene_color, pixel, 0).rgb;
        return vec4<f32>(encode_srgb(saturate(raw)), 1.0);
    }

    let sharp = sample_rcas_hdr(pixel, max_px);
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(size);
    var hdr = sharp + textureSampleLevel(scene_bloom, tone_sampler, uv, 0.0).rgb;
    let centered = (uv * 2.0 - 1.0) * vec2<f32>(0.76, 0.90);
    let vig = 1.0 - dot(centered, centered) * frame_params.grade.w;
    hdr *= frame_params.grade.x * clamp(vig, 0.72, 1.0);

    var mapped = agx_tonemap(hdr);
    let luma = dot(mapped, vec3<f32>(0.2126, 0.7152, 0.0722));
    mapped = saturate(mix(vec3<f32>(luma), mapped, frame_params.grade.z));
    return vec4<f32>(encode_srgb(mapped), 1.0);
}
