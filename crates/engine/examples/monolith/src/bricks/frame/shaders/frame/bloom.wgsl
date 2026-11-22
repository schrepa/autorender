import package::generated::frame::FrameUniforms;

@group(2) @binding(0) var src_tex: texture_2d<f32>;
@group(2) @binding(1) var add_tex: texture_2d<f32>;
@group(2) @binding(2) var bloom_sampler: sampler;
@group(2) @binding(3) var scene_tex: texture_2d<f32>;
@group(2) @binding(4) var<uniform> frame_params: FrameUniforms;

struct BloomVert {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_bloom(@builtin(vertex_index) idx: u32) -> BloomVert {
    let x = f32((idx << 1u) & 2u);
    let y = f32(idx & 2u);
    var out: BloomVert;
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn karis_weight(c: vec3<f32>) -> f32 {
    return 1.0 / (1.0 + luma(c));
}

fn soft_threshold(c: vec3<f32>) -> vec3<f32> {
    let br = max(c.r, max(c.g, c.b));
    let knee = 0.45;
    let thresh = 1.10;
    var soft = br - thresh + knee;
    soft = clamp(soft, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 0.0001);
    let contrib = max(soft, br - thresh) / max(br, 0.0001);
    return max(c * contrib, vec3<f32>(0.0));
}

fn radial_godrays(uv: vec2<f32>, frag_xy: vec2<f32>) -> vec3<f32> {
    if frame_params.fog_params.w > 0.5 {
        return vec3<f32>(0.0);
    }
    let vis = frame_params.sun_uv_vis.z;
    let exposure = frame_params.sun_color_shaft.w;
    if vis * exposure <= 0.005 {
        return vec3<f32>(0.0);
    }
    let sun_uv = frame_params.sun_uv_vis.xy;
    let radial_dist = length((uv - sun_uv) * vec2<f32>(1.30, 1.0));
    let radial_falloff = exp(-radial_dist * 1.28);
    if radial_falloff < 0.01 {
        return vec3<f32>(0.0);
    }
    let density = 0.88;
    let delta = (uv - sun_uv) * (density / 16.0);
    let ign = fract(52.9829189 * fract(dot(frag_xy, vec2<f32>(0.06711056, 0.00583715))));
    var sample_uv = uv - delta * ign;
    var illum_decay = 1.0;
    var accum = vec3<f32>(0.0);
    for (var i = 0; i < 16; i = i + 1) {
        sample_uv -= delta;
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            illum_decay *= 0.8675;
            continue;
        }
        let s = textureSampleLevel(scene_tex, bloom_sampler, sample_uv, 0.0);
        let sky_w = clamp(1.0 - s.a, 0.0, 1.0);
        if sky_w > 0.01 {
            let tint = mix(frame_params.sun_color_shaft.rgb, min(s.rgb, vec3<f32>(3.0)), 0.32);
            accum += tint * (illum_decay * 0.44 * sky_w);
        }
        illum_decay *= 0.8675;
    }
    return accum * (exposure * radial_falloff * vis * (1.0 / 7.95));
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
fn fs_extract(in: BloomVert) -> @location(0) vec4<f32> {
    if frame_params.fog_params.w > 0.5 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let uv = in.uv;
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
    return vec4<f32>(soft_threshold(min(avg, vec3<f32>(16.0))), 1.0);
}

@fragment
fn fs_down(in: BloomVert) -> @location(0) vec4<f32> {
    let src_dim = vec2<f32>(textureDimensions(src_tex));
    let uv = in.uv;
    let d = 1.0 / src_dim;
    var sum = textureSampleLevel(src_tex, bloom_sampler, uv, 0.0).rgb * 4.0;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(-d.x, -d.y), 0.0).rgb;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(d.x, -d.y), 0.0).rgb;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(-d.x, d.y), 0.0).rgb;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(d.x, d.y), 0.0).rgb;
    return vec4<f32>(sum * 0.125, 1.0);
}

@fragment
fn fs_up(in: BloomVert) -> @location(0) vec4<f32> {
    return vec4<f32>(upsample_tent(in.uv), 1.0);
}

@fragment
fn fs_up_final(in: BloomVert) -> @location(0) vec4<f32> {
    let bloom_rgb = upsample_tent(in.uv);
    let shafts = radial_godrays(in.uv, in.pos.xy);
    return vec4<f32>(bloom_rgb * frame_params.grade.y + shafts, 1.0);
}
