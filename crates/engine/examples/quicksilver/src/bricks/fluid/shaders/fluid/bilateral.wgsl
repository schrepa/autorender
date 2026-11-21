import package::fluid::common::FrameUniform;

@group(1) @binding(0) var<uniform> blur_frame: FrameUniform;
@group(1) @binding(1) var depth_in: texture_2d<f32>;
@group(1) @binding(2) var thick_in: texture_2d<f32>;

override FILTER_STEP: f32 = 1.0;

fn bilateral_pass(pos: vec2<f32>, dir: vec2<f32>) -> f32 {
    let dims = vec2<i32>(textureDimensions(depth_in));
    let coord = clamp(vec2<i32>(pos), vec2<i32>(0), dims - 1);
    let z_c = textureLoad(depth_in, coord, 0).r;
    if z_c <= 0.0 {
        return 0.0;
    }

    let r_world = blur_frame.splat.x;
    let focal = blur_frame.splat.y;
    let screen_r = clamp((r_world * 1.85 * focal) / max(z_c, 0.35), 2.0, 14.0);
    var stride = 1.0;
    if FILTER_STEP > 1.05 {
        stride = clamp((screen_r * 0.19) * FILTER_STEP, 1.0, FILTER_STEP * 1.45);
    }

    let sigma_s = 3.5;
    let inv_2s2 = 0.5 / (sigma_s * sigma_s);
    let max_jump = r_world * 5.2;
    let sigma_curv = r_world * 1.85;
    let inv_2c2 = 0.5 / (sigma_curv * sigma_curv);
    let sigma_edge = r_world * 1.35;
    let inv_2e2 = 0.5 / (sigma_edge * sigma_edge);

    var sum = z_c;
    var w_sum = 1.0;

    for (var k = 1; k <= 6; k += 1) {
        let fk = f32(k);
        let ws = exp(-fk * fk * inv_2s2);
        let offset = vec2<i32>(round(dir * (fk * stride)));

        let c_pos = clamp(coord + offset, vec2<i32>(0), dims - 1);
        let c_neg = clamp(coord - offset, vec2<i32>(0), dims - 1);
        let z_pos = textureLoad(depth_in, c_pos, 0).r;
        let z_neg = textureLoad(depth_in, c_neg, 0).r;

        let pos_ok = z_pos > 0.0 && abs(z_pos - z_c) < max_jump;
        let neg_ok = z_neg > 0.0 && abs(z_neg - z_c) < max_jump;

        if pos_ok && neg_ok {
            let curv = 0.5 * (z_pos + z_neg) - z_c;
            let wr = exp(-curv * curv * inv_2c2);
            let w = ws * wr;
            sum += (z_pos + z_neg) * w;
            w_sum += 2.0 * w;
        } else if pos_ok {
            let dz = z_pos - z_c;
            let wr = exp(-dz * dz * inv_2e2);
            let w = ws * wr * 0.65;
            sum += z_pos * w;
            w_sum += w;
        } else if neg_ok {
            let dz = z_neg - z_c;
            let wr = exp(-dz * dz * inv_2e2);
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

fn blur_thick(pos: vec2<f32>, dir: vec2<i32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(thick_in));
    let coord = clamp(vec2<i32>(pos), vec2<i32>(0), dims - 1);
    let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    var acc = textureLoad(thick_in, coord, 0) * weights[0];
    for (var i = 1; i < 5; i += 1) {
        let off = dir * (i * 2);
        acc += textureLoad(thick_in, clamp(coord + off, vec2<i32>(0), dims - 1), 0) * weights[i];
        acc += textureLoad(thick_in, clamp(coord - off, vec2<i32>(0), dims - 1), 0) * weights[i];
    }
    return acc;
}

@fragment
fn fs_thick_h(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return blur_thick(pos.xy, vec2<i32>(1, 0));
}

@fragment
fn fs_thick_v(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return blur_thick(pos.xy, vec2<i32>(0, 1));
}
