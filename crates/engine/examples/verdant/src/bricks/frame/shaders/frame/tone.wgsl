import package::engine::color::encode_srgb;
import package::generated::frame::ToneUniforms;

@group(2) @binding(0) var scene_color: texture_2d<f32>;
@group(2) @binding(1) var scene_bloom: texture_2d<f32>;
@group(2) @binding(2) var scene_depth: texture_depth_2d;
@group(2) @binding(3) var tone_sampler: sampler;
@group(2) @binding(4) var<uniform> tone_params: ToneUniforms;

fn agx_default_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2
        - 40.14 * x4 * x
        + 31.96 * x4
        - 6.868 * x2 * x
        + 0.4298 * x2
        + 0.1191 * x
        - 0.00232;
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
    v = clamp((log2(v) + 12.47393) / 16.5, vec3<f32>(0.0), vec3<f32>(1.0));
    v = agx_default_contrast(v);
    v = max(outset * v, vec3<f32>(0.0));
    return saturate(pow(v, vec3<f32>(2.2)));
}

fn radial_sunbeams(uv: vec2<f32>, depth_size: vec2<i32>) -> vec3<f32> {
    let vis = tone_params.sun_uv_vis.z;
    let shaft_str = tone_params.sun_color_shaft.w;
    if vis <= 0.01 || shaft_str <= 0.01 {
        return vec3<f32>(0.0);
    }
    let sun_uv = tone_params.sun_uv_vis.xy;
    let delta = (uv - sun_uv) * (1.0 / 16.0);
    let dither = fract(sin(dot(uv * 912.3, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    var sample_uv = uv - delta * dither;
    var decay = 1.0;
    var accum = 0.0;
    for (var i = 0; i < 14; i = i + 1) {
        sample_uv -= delta;
        let clamped = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(0.999));
        let px = vec2<i32>(clamped * vec2<f32>(depth_size));
        let d = textureLoad(scene_depth, px, 0);
        let sky_open = select(0.0, 1.0, d <= 0.00001);
        accum += sky_open * decay;
        decay *= 0.89;
    }
    let radial_falloff = exp(-length((uv - sun_uv) * vec2<f32>(1.4, 1.0)) * 2.1);
    return tone_params.sun_color_shaft.rgb * (accum * 0.048 * radial_falloff * vis * shaft_str);
}

@fragment
fn fs_tone(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(scene_color));
    let pixel = min(vec2<i32>(pos.xy), size - 1);
    let uv = (pos.xy) / vec2<f32>(size);

    var hdr = textureLoad(scene_color, pixel, 0).rgb;
    let bloom = textureSampleLevel(scene_bloom, tone_sampler, uv, 0.0).rgb;
    let shafts = radial_sunbeams(uv, size);
    hdr += bloom * tone_params.grade.y + shafts;

    let centered = uv * 2.0 - 1.0;
    let vig = 1.0 - dot(centered * vec2<f32>(0.78, 0.92), centered * vec2<f32>(0.78, 0.92)) * tone_params.grade.w;
    hdr *= tone_params.grade.x * clamp(vig, 0.65, 1.0);

    var mapped = agx_tonemap(hdr);
    let luma = dot(mapped, vec3<f32>(0.2126, 0.7152, 0.0722));
    mapped = saturate(mix(vec3<f32>(luma), mapped, tone_params.grade.z));
    return vec4<f32>(encode_srgb(mapped), 1.0);
}
