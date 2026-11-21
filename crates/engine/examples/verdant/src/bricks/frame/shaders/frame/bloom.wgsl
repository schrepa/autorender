import package::generated::frame::ToneUniforms;

@group(2) @binding(0) var src_tex: texture_2d<f32>;
@group(2) @binding(1) var add_tex: texture_2d<f32>;
@group(2) @binding(2) var bloom_sampler: sampler;
@group(2) @binding(3) var scene_depth: texture_depth_2d;
@group(2) @binding(4) var<uniform> tone_params: ToneUniforms;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn karis_weight(c: vec3<f32>) -> f32 {
    return 1.0 / (1.0 + luma(c));
}

fn soft_threshold(c: vec3<f32>) -> vec3<f32> {
    let br = max(c.r, max(c.g, c.b));
    let knee = 0.55;
    let thresh = 1.45;
    var soft = br - thresh + knee;
    soft = clamp(soft, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 0.0001);
    let contrib = max(soft, br - thresh) / max(br, 0.0001);
    return max(c * contrib, vec3<f32>(0.0));
}

fn radial_sunbeams(uv: vec2<f32>) -> vec3<f32> {
    let vis = tone_params.sun_uv_vis.z;
    let shaft_str = tone_params.sun_color_shaft.w;
    if vis * shaft_str <= 0.015 {
        return vec3<f32>(0.0);
    }
    let sun_uv = tone_params.sun_uv_vis.xy;
    let radial_falloff = exp(-length((uv - sun_uv) * vec2<f32>(1.4, 1.0)) * 2.1);
    if radial_falloff < 0.015 {
        return vec3<f32>(0.0);
    }
    let depth_dim_u = textureDimensions(scene_depth);
    let depth_dim = vec2<f32>(depth_dim_u);
    let depth_max = vec2<i32>(depth_dim_u) - 1;
    let delta = (uv - sun_uv) * (1.0 / 11.5);
    let dither = fract(sin(dot(uv * 912.3, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    var sample_uv = uv - delta * dither;
    var decay = 1.0;
    var accum = 0.0;
    for (var i = 0; i < 10; i = i + 1) {
        sample_uv -= delta;
        let clamped = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(0.999));
        let px = min(vec2<i32>(clamped * depth_dim), depth_max);
        let d = textureLoad(scene_depth, px, 0);
        let sky_open = select(0.0, 1.0, d <= 0.00001);
        accum += sky_open * decay;
        decay *= 0.855;
    }
    return tone_params.sun_color_shaft.rgb * (accum * 0.064 * radial_falloff * vis * shaft_str);
}

fn upsample_tent(uv: vec2<f32>) -> vec3<f32> {
    let d = 0.75 / vec2<f32>(textureDimensions(src_tex));
    var sum = textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(-d.x, -d.y), 0.0).rgb;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(d.x, -d.y), 0.0).rgb;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(-d.x, d.y), 0.0).rgb;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(d.x, d.y), 0.0).rgb;
    let base = textureSampleLevel(add_tex, bloom_sampler, uv, 0.0).rgb;
    return base + sum * (0.25 * 0.85);
}

@fragment
fn fs_extract(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dst_size = vec2<f32>(textureDimensions(src_tex)) * 0.5;
    let uv = pos.xy / max(dst_size, vec2<f32>(1.0));
    let texel = 1.0 / vec2<f32>(textureDimensions(src_tex));
    let c0 = textureSampleLevel(src_tex, bloom_sampler, uv, 0.0).rgb;
    let c1 = textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(-texel.x, -texel.y), 0.0).rgb;
    let c2 = textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(texel.x, -texel.y), 0.0).rgb;
    let c3 = textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(-texel.x, texel.y), 0.0).rgb;
    let c4 = textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(texel.x, texel.y), 0.0).rgb;
    let w0 = karis_weight(c0) * 2.0;
    let w1 = karis_weight(c1);
    let w2 = karis_weight(c2);
    let w3 = karis_weight(c3);
    let w4 = karis_weight(c4);
    let avg = (c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4) / (w0 + w1 + w2 + w3 + w4);
    return vec4<f32>(soft_threshold(min(avg, vec3<f32>(18.0))), 1.0);
}

@fragment
fn fs_down(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let src_dim = vec2<f32>(textureDimensions(src_tex));
    let dst_dim = max(src_dim * 0.5, vec2<f32>(1.0));
    let uv = pos.xy / dst_dim;
    let d = 1.0 / src_dim;
    var sum = textureSampleLevel(src_tex, bloom_sampler, uv, 0.0).rgb * 4.0;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(-d.x, -d.y), 0.0).rgb;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(d.x, -d.y), 0.0).rgb;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(-d.x, d.y), 0.0).rgb;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(d.x, d.y), 0.0).rgb;
    return vec4<f32>(sum * 0.125, 1.0);
}

@fragment
fn fs_up(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let add_dim = vec2<f32>(textureDimensions(add_tex));
    let uv = pos.xy / max(add_dim, vec2<f32>(1.0));
    return vec4<f32>(upsample_tent(uv), 1.0);
}

@fragment
fn fs_up_final(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let add_dim = vec2<f32>(textureDimensions(add_tex));
    let uv = pos.xy / max(add_dim, vec2<f32>(1.0));
    let bloom_rgb = upsample_tent(uv);
    let shafts = radial_sunbeams(uv);
    return vec4<f32>(bloom_rgb * tone_params.grade.y + shafts, 1.0);
}
