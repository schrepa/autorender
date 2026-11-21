import package::fluid::common::FrameUniform;

@group(1) @binding(0) var<uniform> blur_frame: FrameUniform;
@group(1) @binding(1) var depth_in: texture_2d<f32>;
@group(1) @binding(2) var thick_in: texture_2d<f32>;
@group(1) @binding(3) var blur_sampler: sampler;

override FILTER_STEP: f32 = 1.0;

const SPATIAL_W: array<f32, 6> = array<f32, 6>(
    0.9600,
    0.8494,
    0.6926,
    0.5204,
    0.3604,
    0.2301,
);

fn fast_gauss(x: f32) -> f32 {
    let t = 1.0 + x * 0.125;
    let t2 = t * t;
    let t4 = t2 * t2;
    return 1.0 / (t4 * t4);
}

fn bilateral_pass(pos: vec2<f32>, dir: vec2<f32>) -> f32 {
    let dims = vec2<i32>(textureDimensions(depth_in));
    let coord = clamp(vec2<i32>(pos), vec2<i32>(0), dims - 1);
    let z_c = textureLoad(depth_in, coord, 0).r;
    if z_c <= 0.0 {
        return 0.0;
    }

    let r_world = blur_frame.splat.x;
    let depth_scale = f32(dims.y) * blur_frame.viewport.w;
    let focal = blur_frame.splat.y * depth_scale;
    let screen_r = clamp((r_world * 1.85 * focal) / max(z_c, 0.35), 2.0, 14.0);
    var stride = 1.0;
    if FILTER_STEP > 1.05 {
        stride = clamp((screen_r * 0.19) * FILTER_STEP, 1.0, FILTER_STEP * 1.45);
    }

    let max_jump = r_world * 5.2;
    let sigma_curv = r_world * 1.85;
    let inv_2c2 = 0.5 / (sigma_curv * sigma_curv);
    let sigma_edge = r_world * 1.35;
    let inv_2e2 = 0.5 / (sigma_edge * sigma_edge);

    var sum = z_c;
    var w_sum = 1.0;

    for (var k = 1; k <= 6; k += 1) {
        let ws = SPATIAL_W[k - 1];
        let offset = vec2<i32>(round(dir * (f32(k) * stride)));

        let c_pos = clamp(coord + offset, vec2<i32>(0), dims - 1);
        let c_neg = clamp(coord - offset, vec2<i32>(0), dims - 1);
        let z_pos = textureLoad(depth_in, c_pos, 0).r;
        let z_neg = textureLoad(depth_in, c_neg, 0).r;

        let pos_ok = z_pos > 0.0 && abs(z_pos - z_c) < max_jump;
        let neg_ok = z_neg > 0.0 && abs(z_neg - z_c) < max_jump;

        if pos_ok && neg_ok {
            let curv = 0.5 * (z_pos + z_neg) - z_c;
            let wr = fast_gauss(curv * curv * inv_2c2);
            let w = ws * wr;
            sum += (z_pos + z_neg) * w;
            w_sum += 2.0 * w;
        } else if pos_ok {
            let dz = z_pos - z_c;
            let wr = fast_gauss(dz * dz * inv_2e2);
            let w = ws * wr * 0.65;
            sum += z_pos * w;
            w_sum += w;
        } else if neg_ok {
            let dz = z_neg - z_c;
            let wr = fast_gauss(dz * dz * inv_2e2);
            let w = ws * wr * 0.65;
            sum += z_neg * w;
            w_sum += w;
        }
    }

    return sum / max(w_sum, 1.0e-5);
}

@fragment
fn fs_bilateral_h(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
    return bilateral_pass(pos.xy, vec2<f32>(1.0, 0.0));
}

@fragment
fn fs_bilateral_v(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
    return bilateral_pass(pos.xy, vec2<f32>(0.0, 1.0));
}

fn blur_thick(pos: vec2<f32>, dir: vec2<f32>) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(thick_in));
    let uv = pos / dims;
    let step = dir / dims;
    var acc = textureSampleLevel(thick_in, blur_sampler, uv, 0.0) * 0.227027;
    let o1 = step * 2.55;
    let o2 = step * 5.85;
    acc += (
        textureSampleLevel(thick_in, blur_sampler, uv + o1, 0.0)
        + textureSampleLevel(thick_in, blur_sampler, uv - o1, 0.0)
    ) * 0.316216;
    acc += (
        textureSampleLevel(thick_in, blur_sampler, uv + o2, 0.0)
        + textureSampleLevel(thick_in, blur_sampler, uv - o2, 0.0)
    ) * 0.070270;
    return acc;
}

@fragment
fn fs_thick_h(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return blur_thick(pos.xy, vec2<f32>(1.0, 0.0));
}

@fragment
fn fs_thick_v(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return blur_thick(pos.xy, vec2<f32>(0.0, 1.0));
}
