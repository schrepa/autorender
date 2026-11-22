import package::generated::frame::FrameUniforms;

@group(2) @binding(0) var toned_tex: texture_2d<f32>;
@group(2) @binding(1) var fxaa_sampler: sampler;
@group(2) @binding(2) var<uniform> frame_params: FrameUniforms;

fn sample_luma(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(toned_tex, fxaa_sampler, uv, 0.0).a;
}

@fragment
fn fs_fxaa(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dim = vec2<f32>(textureDimensions(toned_tex));
    let inv = 1.0 / max(dim, vec2<f32>(1.0));
    let uv = pos.xy * inv;

    let center = textureSampleLevel(toned_tex, fxaa_sampler, uv, 0.0);
    if frame_params.fog_params.w > 0.5 {
        return vec4<f32>(center.rgb, 1.0);
    }

    let lm = center.a;
    let ln = sample_luma(uv + vec2<f32>(0.0, -inv.y));
    let ls = sample_luma(uv + vec2<f32>(0.0, inv.y));
    let lw = sample_luma(uv + vec2<f32>(-inv.x, 0.0));
    let le = sample_luma(uv + vec2<f32>(inv.x, 0.0));

    let lmin = min(lm, min(min(ln, ls), min(lw, le)));
    let lmax = max(lm, max(max(ln, ls), max(lw, le)));
    let range = lmax - lmin;
    if range < max(0.0312, lmax * 0.0625) {
        return vec4<f32>(center.rgb, 1.0);
    }

    let lnw = sample_luma(uv + vec2<f32>(-inv.x, -inv.y));
    let lne = sample_luma(uv + vec2<f32>(inv.x, -inv.y));
    let lsw = sample_luma(uv + vec2<f32>(-inv.x, inv.y));
    let lse = sample_luma(uv + vec2<f32>(inv.x, inv.y));

    let lns = ln + ls;
    let lwe = lw + le;
    let subpix_raw = clamp(
        abs((2.0 * (lns + lwe) + (lnw + lne + lsw + lse)) * (1.0 / 12.0) - lm) / range,
        0.0,
        1.0,
    );
    let subpix_s = smoothstep(0.0, 1.0, subpix_raw);
    let subpix_blend = subpix_s * subpix_s * 0.75;

    let edge_h = abs(lnw + lne - 2.0 * ln)
        + 2.0 * abs(lw + le - 2.0 * lm)
        + abs(lsw + lse - 2.0 * ls);
    let edge_v = abs(lnw + lsw - 2.0 * lw)
        + 2.0 * abs(ln + ls - 2.0 * lm)
        + abs(lne + lse - 2.0 * le);
    let is_horiz = edge_h >= edge_v;

    let luma1 = select(lw, ln, is_horiz);
    let luma2 = select(le, ls, is_horiz);
    let grad1 = abs(luma1 - lm);
    let grad2 = abs(luma2 - lm);
    let is_1_steepest = grad1 >= grad2;
    let grad_scaled = 0.25 * max(grad1, grad2);

    var step_len = select(inv.x, inv.y, is_horiz);
    var luma_local_avg = 0.0;
    if is_1_steepest {
        step_len = -step_len;
        luma_local_avg = 0.5 * (luma1 + lm);
    } else {
        luma_local_avg = 0.5 * (luma2 + lm);
    }

    var cur_uv = uv;
    if is_horiz {
        cur_uv.y += step_len * 0.5;
    } else {
        cur_uv.x += step_len * 0.5;
    }
    let off = select(vec2<f32>(0.0, inv.y), vec2<f32>(inv.x, 0.0), is_horiz);
    var uv1 = cur_uv - off;
    var uv2 = cur_uv + off;
    var end1 = sample_luma(uv1) - luma_local_avg;
    var end2 = sample_luma(uv2) - luma_local_avg;
    var reached1 = abs(end1) >= grad_scaled;
    var reached2 = abs(end2) >= grad_scaled;

    if !reached1 {
        uv1 -= off;
    }
    if !reached2 {
        uv2 += off;
    }

    if !(reached1 && reached2) {
        for (var i = 0; i < 8; i = i + 1) {
            var stride = 1.5;
            if i == 1 || i == 2 {
                stride = 2.0;
            } else if i == 3 || i == 4 {
                stride = 2.0;
            } else if i == 5 {
                stride = 4.0;
            } else if i >= 6 {
                stride = 8.0;
            }
            if !reached1 {
                end1 = sample_luma(uv1) - luma_local_avg;
                reached1 = abs(end1) >= grad_scaled;
            }
            if !reached2 {
                end2 = sample_luma(uv2) - luma_local_avg;
                reached2 = abs(end2) >= grad_scaled;
            }
            if reached1 && reached2 {
                break;
            }
            if !reached1 {
                uv1 -= off * stride;
            }
            if !reached2 {
                uv2 += off * stride;
            }
        }
    }

    let dist1 = select(uv.y - uv1.y, uv.x - uv1.x, is_horiz);
    let dist2 = select(uv2.y - uv.y, uv2.x - uv.x, is_horiz);
    let is_dir1 = dist1 < dist2;
    let dist_min = min(dist1, dist2);
    let edge_span = max(dist1 + dist2, 1.0e-5);
    let pixel_offset = -dist_min / edge_span + 0.5;
    let is_center_smaller = lm < luma_local_avg;
    let end_near = select(end2, end1, is_dir1);
    let correct_var = (end_near < 0.0) != is_center_smaller;
    let edge_blend = select(0.0, pixel_offset, correct_var);
    let final_blend = max(edge_blend, subpix_blend);

    var final_uv = uv;
    if is_horiz {
        final_uv.y += final_blend * step_len;
    } else {
        final_uv.x += final_blend * step_len;
    }
    let aa_rgb = textureSampleLevel(toned_tex, fxaa_sampler, final_uv, 0.0).rgb;
    return vec4<f32>(aa_rgb, 1.0);
}
