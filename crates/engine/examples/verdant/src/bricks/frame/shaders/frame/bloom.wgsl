@group(2) @binding(0) var src_tex: texture_2d<f32>;
@group(2) @binding(1) var add_tex: texture_2d<f32>;
@group(2) @binding(2) var bloom_sampler: sampler;

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
    let d = 1.25 / vec2<f32>(textureDimensions(src_tex));
    var sum = textureSampleLevel(src_tex, bloom_sampler, uv, 0.0).rgb * 4.0;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(-d.x, 0.0), 0.0).rgb * 2.0;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(d.x, 0.0), 0.0).rgb * 2.0;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(0.0, -d.y), 0.0).rgb * 2.0;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(0.0, d.y), 0.0).rgb * 2.0;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(-d.x, -d.y), 0.0).rgb;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(d.x, -d.y), 0.0).rgb;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(-d.x, d.y), 0.0).rgb;
    sum += textureSampleLevel(src_tex, bloom_sampler, uv + vec2<f32>(d.x, d.y), 0.0).rgb;
    let base = textureSampleLevel(add_tex, bloom_sampler, uv, 0.0).rgb;
    return vec4<f32>(base + sum * (1.0 / 16.0) * 0.85, 1.0);
}
