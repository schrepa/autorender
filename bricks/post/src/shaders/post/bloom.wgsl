import package::generated::post::PostParams;
import package::post::tonemap::POST_FLAG_DISABLE_BLOOM;
import package::post::tonemap::POST_FLAG_DISABLE_SHAFTS;
import package::post::tonemap::POST_FLAG_STANDARD_Z;
import package::post::tonemap::interleaved_gradient_noise;
import package::post::tonemap::post_luma;

@group(1) @binding(0) var<uniform> post_params: PostParams;
@group(1) @binding(1) var hdr_src: texture_2d<f32>;
@group(1) @binding(2) var depth_tex: texture_depth_2d;
@group(1) @binding(3) var bloom_src: texture_2d<f32>;
@group(1) @binding(4) var bloom_add: texture_2d<f32>;
@group(1) @binding(5) var bloom_sampler: sampler;
@group(1) @binding(6) var bloom_dst: texture_storage_2d<rgba16float, write>;

fn sanitize_hdr(c: vec3<f32>) -> vec3<f32> {
    return select(vec3<f32>(0.0), clamp(c, vec3<f32>(0.0), vec3<f32>(64.0)), c == c);
}

fn karis_weight(c: vec3<f32>) -> f32 {
    return 1.0 / (1.0 + post_luma(c));
}

fn soft_threshold(c: vec3<f32>, threshold: f32, knee_in: f32) -> vec3<f32> {
    let br = max(c.r, max(c.g, c.b));
    let thresh = max(threshold, 0.0);
    let knee = max(knee_in, 1.0e-4);
    var soft = br - thresh + knee;
    soft = clamp(soft, 0.0, 2.0 * knee);
    soft = (soft * soft) / (4.0 * knee + 1.0e-5);
    let contrib = max(soft, br - thresh) / max(br, 1.0e-5);
    return max(c * contrib, vec3<f32>(0.0));
}

fn prefilter_hdr(uv: vec2<f32>, dst_dim: vec2<f32>) -> vec3<f32> {
    if (post_params.flags & POST_FLAG_DISABLE_BLOOM) != 0u || post_params.bloom_intensity <= 0.0 {
        return vec3<f32>(0.0);
    }
    let quad = 0.25 / dst_dim;
    let c0 = sanitize_hdr(textureSampleLevel(hdr_src, bloom_sampler, uv + vec2<f32>(-quad.x, -quad.y), 0.0).rgb);
    let c1 = sanitize_hdr(textureSampleLevel(hdr_src, bloom_sampler, uv + vec2<f32>(quad.x, -quad.y), 0.0).rgb);
    let c2 = sanitize_hdr(textureSampleLevel(hdr_src, bloom_sampler, uv + vec2<f32>(-quad.x, quad.y), 0.0).rgb);
    let c3 = sanitize_hdr(textureSampleLevel(hdr_src, bloom_sampler, uv + vec2<f32>(quad.x, quad.y), 0.0).rgb);
    let w0 = karis_weight(c0);
    let w1 = karis_weight(c1);
    let w2 = karis_weight(c2);
    let w3 = karis_weight(c3);
    let avg = (c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3) / max(w0 + w1 + w2 + w3, 1.0e-5);
    let extracted = soft_threshold(avg, post_params.bloom_threshold, post_params.bloom_knee);
    return extracted * post_params.bloom_intensity;
}

fn radial_sunbeams(uv: vec2<f32>, pixel: vec2<u32>) -> vec3<f32> {
    if (post_params.flags & POST_FLAG_DISABLE_SHAFTS) != 0u
        || post_params.shaft_intensity <= 0.0
        || post_params.sun_visibility <= 0.0
    {
        return vec3<f32>(0.0);
    }
    let depth_dim_u = textureDimensions(depth_tex);
    if depth_dim_u.x == 0u || depth_dim_u.y == 0u {
        return vec3<f32>(0.0);
    }
    let depth_dim = vec2<f32>(depth_dim_u);
    let aspect = depth_dim.x / max(depth_dim.y, 1.0);
    let to_sun = (uv - post_params.sun_uv) * vec2<f32>(aspect, 1.0);
    let radial_falloff = exp(-length(to_sun) * 1.9);
    if radial_falloff < 0.01 {
        return vec3<f32>(0.0);
    }
    let depth_max = max(vec2<i32>(depth_dim_u) - vec2<i32>(1, 1), vec2<i32>(0, 0));
    let step_uv = (uv - post_params.sun_uv) * (1.0 / 12.5);
    let dither = interleaved_gradient_noise(vec2<f32>(pixel), post_params.time);
    var sample_uv = uv - step_uv * dither;
    var decay = 1.0;
    var accum = 0.0;
    let standard_z = (post_params.flags & POST_FLAG_STANDARD_Z) != 0u;
    for (var i: i32 = 0; i < 12; i = i + 1) {
        sample_uv = sample_uv - step_uv;
        let in_bounds = all(sample_uv >= vec2<f32>(0.0)) && all(sample_uv <= vec2<f32>(1.0));
        let clamped = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(0.9999));
        let px = min(vec2<i32>(clamped * depth_dim), depth_max);
        let d = textureLoad(depth_tex, px, 0);
        let is_sky = select(d <= 1.0e-5, d >= 0.99999, standard_z);
        let open = select(0.0, 1.0, in_bounds && is_sky);
        accum = accum + open * decay;
        decay = decay * 0.86;
    }
    let norm = 0.075 * radial_falloff * post_params.sun_visibility * post_params.shaft_intensity;
    return max(post_params.shaft_color, vec3<f32>(0.0)) * (accum * norm);
}

fn sample_bloom(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(bloom_src, bloom_sampler, uv, 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn cs_bloom_prefilter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(bloom_dst);
    if gid.x >= dst_size.x || gid.y >= dst_size.y {
        return;
    }
    let dst_dim = max(vec2<f32>(dst_size), vec2<f32>(1.0));
    let uv = (vec2<f32>(gid.xy) + 0.5) / dst_dim;
    let out_rgb = prefilter_hdr(uv, dst_dim);
    textureStore(bloom_dst, vec2<i32>(gid.xy), vec4<f32>(out_rgb, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_bloom_prefilter_shafts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(bloom_dst);
    if gid.x >= dst_size.x || gid.y >= dst_size.y {
        return;
    }
    let dst_dim = max(vec2<f32>(dst_size), vec2<f32>(1.0));
    let uv = (vec2<f32>(gid.xy) + 0.5) / dst_dim;
    let out_rgb = prefilter_hdr(uv, dst_dim) + radial_sunbeams(uv, gid.xy);
    textureStore(bloom_dst, vec2<i32>(gid.xy), vec4<f32>(out_rgb, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_bloom_downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(bloom_dst);
    if gid.x >= dst_size.x || gid.y >= dst_size.y {
        return;
    }
    let dst_dim = max(vec2<f32>(dst_size), vec2<f32>(1.0));
    let src_dim = max(vec2<f32>(textureDimensions(bloom_src)), vec2<f32>(1.0));
    let uv = (vec2<f32>(gid.xy) + 0.5) / dst_dim;
    let texel = 1.0 / src_dim;
    let a = sample_bloom(uv + vec2<f32>(-2.0, -2.0) * texel);
    let b = sample_bloom(uv + vec2<f32>(0.0, -2.0) * texel);
    let c = sample_bloom(uv + vec2<f32>(2.0, -2.0) * texel);
    let d = sample_bloom(uv + vec2<f32>(-1.0, -1.0) * texel);
    let e = sample_bloom(uv + vec2<f32>(1.0, -1.0) * texel);
    let f = sample_bloom(uv + vec2<f32>(-2.0, 0.0) * texel);
    let g = sample_bloom(uv);
    let h = sample_bloom(uv + vec2<f32>(2.0, 0.0) * texel);
    let i = sample_bloom(uv + vec2<f32>(-1.0, 1.0) * texel);
    let j = sample_bloom(uv + vec2<f32>(1.0, 1.0) * texel);
    let k = sample_bloom(uv + vec2<f32>(-2.0, 2.0) * texel);
    let l = sample_bloom(uv + vec2<f32>(0.0, 2.0) * texel);
    let m = sample_bloom(uv + vec2<f32>(2.0, 2.0) * texel);
    let sum = g * 0.125
        + (d + e + i + j) * 0.125
        + (b + f + h + l) * 0.0625
        + (a + c + k + m) * 0.03125;
    textureStore(bloom_dst, vec2<i32>(gid.xy), vec4<f32>(max(sum, vec3<f32>(0.0)), 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_bloom_upsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(bloom_dst);
    if gid.x >= dst_size.x || gid.y >= dst_size.y {
        return;
    }
    let dst_dim = max(vec2<f32>(dst_size), vec2<f32>(1.0));
    let src_dim = max(vec2<f32>(textureDimensions(bloom_src)), vec2<f32>(1.0));
    let uv = (vec2<f32>(gid.xy) + 0.5) / dst_dim;
    let texel = 1.0 / src_dim;
    let c00 = sample_bloom(uv + vec2<f32>(-texel.x, -texel.y));
    let c10 = sample_bloom(uv + vec2<f32>(0.0, -texel.y));
    let c20 = sample_bloom(uv + vec2<f32>(texel.x, -texel.y));
    let c01 = sample_bloom(uv + vec2<f32>(-texel.x, 0.0));
    let c11 = sample_bloom(uv);
    let c21 = sample_bloom(uv + vec2<f32>(texel.x, 0.0));
    let c02 = sample_bloom(uv + vec2<f32>(-texel.x, texel.y));
    let c12 = sample_bloom(uv + vec2<f32>(0.0, texel.y));
    let c22 = sample_bloom(uv + vec2<f32>(texel.x, texel.y));
    let tent = (c11 * 4.0 + (c10 + c01 + c21 + c12) * 2.0 + (c00 + c20 + c02 + c22)) * 0.0625;
    let add_max = max(vec2<i32>(textureDimensions(bloom_add)) - vec2<i32>(1, 1), vec2<i32>(0, 0));
    let add_coord = min(vec2<i32>(gid.xy), add_max);
    let base = textureLoad(bloom_add, add_coord, 0).rgb;
    let out_rgb = max(mix(base, tent, 0.55), vec3<f32>(0.0));
    textureStore(bloom_dst, vec2<i32>(gid.xy), vec4<f32>(out_rgb, 1.0));
}
