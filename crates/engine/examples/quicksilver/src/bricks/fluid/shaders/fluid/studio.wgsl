import package::fluid::common::FrameUniform;

@group(1) @binding(0) var<uniform> studio_frame: FrameUniform;
@group(1) @binding(1) var studio_depth: texture_2d<f32>;
@group(1) @binding(2) var studio_aux: texture_2d<f32>;
@group(1) @binding(3) var studio_thick: texture_2d<f32>;
@group(1) @binding(4) var studio_sampler: sampler;
@group(1) @binding(5) var<storage, read> studio_heightfield: array<vec4<f32>>;

fn hash21(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(123.34, 456.21));
    let r = q + dot(q, q + 45.32);
    return fract(r.x * r.y);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm2(p: vec2<f32>) -> f32 {
    var v = value_noise(p) * 0.56;
    let q = mat2x2<f32>(1.6, 1.2, -1.2, 1.6) * p;
    v += value_noise(q) * 0.29;
    let q2 = mat2x2<f32>(1.6, 1.2, -1.2, 1.6) * q;
    v += value_noise(q2) * 0.15;
    return v;
}

fn rect_light(dir: vec3<f32>, center: vec3<f32>, u_axis: vec3<f32>, v_axis: vec3<f32>, half_uv: vec2<f32>, soft: f32) -> f32 {
    let d_dot = dot(dir, center);
    if d_dot <= 0.05 {
        return 0.0;
    }
    let proj = dir / d_dot - center;
    let u = abs(dot(proj, u_axis)) - half_uv.x;
    let v = abs(dot(proj, v_axis)) - half_uv.y;
    let dist = length(max(vec2<f32>(u, v), vec2<f32>(0.0))) + min(max(u, v), 0.0);
    return 1.0 - smoothstep(-soft, soft, dist);
}

fn sample_studio_hdri(dir: vec3<f32>, mat_id: f32) -> vec3<f32> {
    let d = normalize(dir);
    let az = atan2(d.z, d.x);

    let sky_band = smoothstep(-0.08, 0.42, d.y) * (1.0 - smoothstep(0.68, 0.98, d.y));
    let dome_cyc = mix(
        vec3<f32>(0.055, 0.064, 0.082),
        vec3<f32>(0.36, 0.41, 0.50),
        sky_band,
    );
    let zenith_cap = vec3<f32>(0.18, 0.21, 0.27) * smoothstep(0.55, 1.0, d.y);
    var base = dome_cyc + zenith_cap;

    let studio_bays = smoothstep(-0.15, 0.35, cos(az * 2.0 - 0.55));
    let neg_flag_1 = exp(-pow(sin((az + 1.15) * 0.5), 2.0) * 14.0);
    let neg_flag_2 = exp(-pow(sin((az - 2.05) * 0.5), 2.0) * 16.0);
    let neg_fill = clamp(neg_flag_1 + neg_flag_2, 0.0, 0.92);
    base *= (0.45 + 0.72 * studio_bays) * (1.0 - 0.86 * neg_fill);

    let dark_equator = exp(-pow((d.y + 0.07) * 9.5, 2.0));
    base *= (1.0 - 0.78 * dark_equator);

    let horizon_strip = exp(-pow((d.y - 0.06) * 13.0, 2.0));
    let strip_panels = smoothstep(0.0, 0.42, cos(az * 2.0 - 0.5)) * (1.0 - neg_fill);
    base += vec3<f32>(1.15, 1.26, 1.45) * horizon_strip * strip_panels;

    let ground_bounce = vec3<f32>(0.045, 0.052, 0.068) * smoothstep(0.0, -0.55, d.y);
    base = mix(base, ground_bounce, smoothstep(-0.02, -0.22, d.y));

    let k1 = rect_light(
        d,
        normalize(vec3<f32>(-0.28, 0.88, 0.38)),
        normalize(vec3<f32>(0.91, 0.16, 0.38)),
        normalize(vec3<f32>(-0.20, -0.42, 0.88)),
        vec2<f32>(0.72, 0.20),
        0.09,
    );
    let k2 = rect_light(
        d,
        normalize(vec3<f32>(0.54, 0.70, -0.46)),
        normalize(vec3<f32>(0.42, 0.12, 0.90)),
        normalize(vec3<f32>(-0.70, 0.70, 0.12)),
        vec2<f32>(0.56, 0.12),
        0.055,
    );
    let k3 = rect_light(
        d,
        normalize(vec3<f32>(0.80, 0.24, 0.54)),
        normalize(vec3<f32>(-0.56, 0.0, 0.83)),
        vec3<f32>(0.0, 1.0, 0.0),
        vec2<f32>(0.36, 0.09),
        0.06,
    );
    let k4 = rect_light(
        d,
        normalize(vec3<f32>(-0.82, 0.26, -0.50)),
        normalize(vec3<f32>(0.52, 0.0, -0.85)),
        vec3<f32>(0.0, 1.0, 0.0),
        vec2<f32>(0.44, 0.09),
        0.065,
    );

    var env = base;
    env += vec3<f32>(4.6, 4.8, 5.2) * k1;
    env += vec3<f32>(3.6, 4.1, 4.9) * k2;
    env += vec3<f32>(2.9, 1.82, 1.02) * k3;
    env += vec3<f32>(1.15, 1.72, 2.75) * k4;

    if mat_id > 2.5 {
        env *= 0.28;
    }
    return env;
}

fn sample_studio_bg(dir: vec3<f32>) -> vec3<f32> {
    let d = normalize(dir);
    let up = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
    var col = mix(vec3<f32>(0.009, 0.012, 0.018), vec3<f32>(0.030, 0.037, 0.050), smoothstep(0.0, 0.52, up));
    col = mix(col, vec3<f32>(0.013, 0.017, 0.025), smoothstep(0.52, 1.0, up));
    let glow = exp(-pow((d.y - 0.14) * 5.0, 2.0)) * 0.020;
    return col + vec3<f32>(0.65, 0.78, 1.0) * glow;
}

fn sample_heightfield(wx: f32, wz: f32) -> vec4<f32> {
    let cs = studio_frame.basin.w;
    let gx_f = wx / cs + studio_frame.grid_dims.x * 0.5;
    let gz_f = wz / cs + studio_frame.grid_dims.z * 0.5;
    let gx_max = i32(studio_frame.grid_dims.x) - 1;
    let gz_max = i32(studio_frame.grid_dims.z) - 1;
    if gx_f < 0.0 || gz_f < 0.0 || gx_f > f32(gx_max) || gz_f > f32(gz_max) {
        return vec4<f32>(0.0);
    }
    let ix = clamp(i32(floor(gx_f)), 0, gx_max - 1);
    let iz = clamp(i32(floor(gz_f)), 0, gz_max - 1);
    let fx = clamp(gx_f - f32(ix), 0.0, 1.0);
    let fz = clamp(gz_f - f32(iz), 0.0, 1.0);
    let stride = u32(studio_frame.grid_dims.x);

    let v00 = studio_heightfield[u32(iz) * stride + u32(ix)];
    let v10 = studio_heightfield[u32(iz) * stride + u32(ix + 1)];
    let v01 = studio_heightfield[u32(iz + 1) * stride + u32(ix)];
    let v11 = studio_heightfield[u32(iz + 1) * stride + u32(ix + 1)];
    return mix(mix(v00, v10, fx), mix(v01, v11, fx), fz);
}

fn caustic_pattern(p: vec2<f32>, t: f32) -> f32 {
    var uv = p * 2.65;
    var acc = vec2<f32>(0.0);
    let rot = mat2x2<f32>(0.80, -0.60, 0.60, 0.80);
    for (var i = 0; i < 3; i += 1) {
        let fi = f32(i);
        let phase = t * (0.95 + fi * 0.28);
        let s = sin(uv.x + cos(uv.y + phase)) + cos(uv.y - sin(uv.x - phase * 0.8));
        acc += vec2<f32>(cos(s * 1.45), sin(s * 1.45));
        uv = rot * uv * 1.38 + vec2<f32>(0.7, -0.4);
    }
    let mag = length(acc) / 3.0;
    let web = 1.0 - clamp(mag, 0.0, 1.0);
    return pow(web, 3.5) * 2.25;
}

fn eval_floor_caustics(wx: f32, wz: f32, mat_id: f32) -> vec4<f32> {
    let eps = 0.095;
    let hc = sample_heightfield(wx, wz);
    let h_xp = sample_heightfield(wx + eps, wz);
    let h_xm = sample_heightfield(wx - eps, wz);
    let h_zp = sample_heightfield(wx, wz + eps);
    let h_zm = sample_heightfield(wx, wz - eps);

    let eps_sh = 0.24;
    let s1 = sample_heightfield(wx + eps_sh, wz + eps_sh).y;
    let s2 = sample_heightfield(wx - eps_sh, wz + eps_sh).y;
    let s3 = sample_heightfield(wx + eps_sh, wz - eps_sh).y;
    let s4 = sample_heightfield(wx - eps_sh, wz - eps_sh).y;
    let col_mass = (hc.y * 2.0 + h_xp.y + h_xm.y + h_zp.y + h_zm.y) / 6.0;
    let soft_mass = (col_mass * 2.0 + s1 + s2 + s3 + s4) / 6.0;

    let coverage = smoothstep(0.6, 5.5, col_mass);
    let lap = (h_xp.x + h_xm.x + h_zp.x + h_zm.x - 4.0 * hc.x);
    let grad = vec2<f32>(h_xp.x - h_xm.x, h_zp.x - h_zm.x) / (2.0 * eps);
    let focus = clamp(-lap * 0.68 + length(grad) * 0.17, 0.0, 2.5);

    let t = studio_frame.eye_time.w;
    let shadow = clamp(1.0 - smoothstep(1.2, 48.0, soft_mass) * 0.62, 0.34, 1.0);
    let is_zero_g = studio_frame.style.y > 0.5;

    if mat_id > 0.5 && mat_id < 1.5 {
        if is_zero_g {
            return vec4<f32>(vec3<f32>(0.02, 0.14, 0.18) * smoothstep(2.0, 28.0, soft_mass) * 0.35, shadow);
        }
        let warp = vec2<f32>(wx, wz) + grad * 0.068;
        let cr = caustic_pattern(warp + vec2<f32>(0.018, 0.008), t);
        let cg = caustic_pattern(warp, t);
        let cb = caustic_pattern(warp - vec2<f32>(0.018, 0.010), t);
        let dyn = (0.52 + focus * 1.25) * (0.88 + 0.35 * clamp(hc.z, 0.0, 1.5)) * coverage;
        let rgb = vec3<f32>(cr, cg, cb) * vec3<f32>(0.84, 1.06, 1.18) * dyn * 1.32;
        return vec4<f32>(rgb, shadow);
    }

    if mat_id > 1.5 && mat_id < 2.5 {
        let glow = smoothstep(1.0, 18.0, soft_mass) * (0.35 + 0.45 * clamp(hc.z * 0.7, 0.0, 1.2));
        return vec4<f32>(vec3<f32>(1.15, 0.42, 0.05) * glow * 0.42, shadow);
    }

    if mat_id > 2.5 {
        let glow = smoothstep(0.8, 16.0, soft_mass) * (0.42 + 0.45 * clamp(hc.z * 0.8, 0.0, 1.3));
        let shift = 0.5 + 0.5 * sin(wx * 1.5 + wz * 1.3 - t * 1.7);
        let tint = mix(vec3<f32>(0.05, 0.78, 1.20), vec3<f32>(0.72, 0.16, 1.18), shift);
        return vec4<f32>(tint * glow * 0.48, shadow);
    }

    return vec4<f32>(vec3<f32>(0.0), shadow);
}

struct Hit {
    t: f32,
    normal: vec3<f32>,
    kind: u32,
}

fn intersect_box(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec4<f32> {
    let inv = 1.0 / select(rd, vec3<f32>(1.0e-6), abs(rd) < vec3<f32>(1.0e-6));
    let t0 = (bmin - ro) * inv;
    let t1 = (bmax - ro) * inv;
    let tsmall = min(t0, t1);
    let tbig = max(t0, t1);
    let t_enter = max(max(tsmall.x, tsmall.y), tsmall.z);
    let t_exit = min(min(tbig.x, tbig.y), tbig.z);
    if t_exit < max(t_enter, 0.0) {
        return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    }
    var n = vec3<f32>(0.0);
    if t_enter == tsmall.x {
        n.x = -sign(rd.x);
    } else if t_enter == tsmall.y {
        n.y = -sign(rd.y);
    } else {
        n.z = -sign(rd.z);
    }
    return vec4<f32>(t_enter, n);
}

fn trace_studio(ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    var best = Hit(1.0e9, vec3<f32>(0.0, 1.0, 0.0), 0u);
    let hx = studio_frame.basin.x;
    let hz = studio_frame.basin.y;
    let fy = studio_frame.basin.z;
    let rim_h = 0.17;
    let rim_w = 0.14;
    let plinth_y = -0.24;

    if rd.y < -1.0e-5 {
        let t_floor = (plinth_y - ro.y) / rd.y;
        if t_floor > 0.0 && t_floor < best.t {
            best = Hit(t_floor, vec3<f32>(0.0, 1.0, 0.0), 1u);
        }
        let t_basin = (fy - ro.y) / rd.y;
        if t_basin > 0.0 && t_basin < best.t {
            let p = ro + rd * t_basin;
            if abs(p.x) <= hx + 0.02 && abs(p.z) <= hz + 0.02 {
                best = Hit(t_basin, vec3<f32>(0.0, 1.0, 0.0), 2u);
            }
        }
    }

    let walls = array<vec3<f32>, 8>(
        vec3<f32>(-(hx + rim_w), plinth_y, -(hz + rim_w)),
        vec3<f32>(-hx, fy + rim_h, hz + rim_w),
        vec3<f32>(hx, plinth_y, -(hz + rim_w)),
        vec3<f32>(hx + rim_w, fy + rim_h, hz + rim_w),
        vec3<f32>(-hx, plinth_y, -(hz + rim_w)),
        vec3<f32>(hx, fy + rim_h, -hz),
        vec3<f32>(-hx, plinth_y, hz),
        vec3<f32>(hx, fy + rim_h, hz + rim_w),
    );
    for (var i = 0; i < 4; i += 1) {
        let h = intersect_box(ro, rd, walls[i * 2], walls[i * 2 + 1]);
        if h.x > 0.0 && h.x < best.t {
            best = Hit(h.x, h.yzw, 3u);
        }
    }

    let base_hit = intersect_box(
        ro,
        rd,
        vec3<f32>(-(hx + rim_w), plinth_y, -(hz + rim_w)),
        vec3<f32>(hx + rim_w, fy, hz + rim_w),
    );
    if base_hit.x > 0.0 && base_hit.x < best.t {
        best = Hit(base_hit.x, base_hit.yzw, 3u);
    }

    return best;
}

fn shade_studio_hit(hit: Hit, p: vec3<f32>, rd: vec3<f32>, mat_id: f32) -> vec3<f32> {
    let hx = studio_frame.basin.x;
    let hz = studio_frame.basin.y;
    let fy = studio_frame.basin.z;
    let rim_h = 0.17;
    let rim_w = 0.14;
    let v = -rd;
    let n = hit.normal;
    let key_dir = normalize(vec3<f32>(-0.28, 0.88, 0.38));
    let rim_dir = normalize(vec3<f32>(0.54, 0.70, -0.46));

    if hit.kind == 2u {
        let tile = abs(fract(p.xz * 1.45 + 0.5) - 0.5);
        let grout = 1.0 - smoothstep(0.478, 0.494, max(tile.x, tile.y));
        let grain = fbm2(p.xz * 3.6);
        let vein = pow(1.0 - abs(value_noise(p.xz * 2.8 + vec2<f32>(grain * 1.3, -grain)) * 2.0 - 1.0), 6.0);

        var albedo = vec3<f32>(0.022, 0.026, 0.033) * (0.86 + 0.28 * grain) + vec3<f32>(0.013, 0.015, 0.020) * vein;
        if mat_id > 0.5 && mat_id < 1.5 {
            albedo = vec3<f32>(0.044, 0.068, 0.084) * (0.84 + 0.30 * grain) + vec3<f32>(0.020, 0.032, 0.040) * vein;
        }
        albedo *= (0.72 + 0.28 * grout);

        let edge_dist = min(hx - abs(p.x), hz - abs(p.z));
        let wall_ao = smoothstep(0.0, 0.34, edge_dist);
        let caust = eval_floor_caustics(p.x, p.z, mat_id);
        let ndl = max(dot(n, key_dir), 0.0);
        let ndr = max(dot(n, rim_dir), 0.0);

        let r_dir = reflect(rd, n);
        let spec_env = sample_studio_bg(r_dir);
        let fres = 0.025 + 0.28 * pow(1.0 - max(dot(n, v), 0.0), 5.0);

        var col = albedo * (0.28 + 1.15 * ndl + 0.30 * ndr) * (0.50 + 0.50 * wall_ao) * caust.w;
        col += spec_env * fres * 0.45 * wall_ao * caust.w;
        col += caust.rgb * (0.62 + 0.38 * wall_ao);
        return col;
    }

    if hit.kind == 3u {
        let rim_top = smoothstep(fy + rim_h - 0.015, fy + rim_h, p.y);
        let inner_d = max(abs(p.x) - hx, abs(p.z) - hz);
        let outer_d = max(abs(p.x) - (hx + rim_w), abs(p.z) - (hz + rim_w));
        let top_bevel = rim_top * (
            (1.0 - smoothstep(0.0, 0.016, abs(inner_d)))
            + (1.0 - smoothstep(0.0, 0.016, abs(outer_d)))
        );
        let inlay = rim_top * (1.0 - smoothstep(0.008, 0.018, abs(inner_d - 0.032)));
        let grain = value_noise(p.xz * 8.0 + vec2<f32>(p.y * 6.0, 0.0));
        let obsidian = vec3<f32>(0.016, 0.019, 0.024) * (0.88 + 0.24 * grain);
        let metal_col = select(vec3<f32>(0.75, 0.80, 0.88), vec3<f32>(0.98, 0.76, 0.32), mat_id > 1.5 && mat_id < 2.5);

        let ndl = max(dot(n, key_dir), 0.0);
        let ndr = max(dot(n, rim_dir), 0.0);
        let h_key = normalize(key_dir + v);
        let h_rim = normalize(rim_dir + v);
        let spec_rough = pow(max(dot(n, h_key), 0.0), 42.0) * 0.28 + pow(max(dot(n, h_rim), 0.0), 32.0) * 0.16;
        let fres = 0.03 + 0.25 * pow(1.0 - max(dot(n, v), 0.0), 5.0);

        let bottom_ao = smoothstep(-0.24, -0.05, p.y);
        var col = obsidian * (0.24 + 0.95 * ndl + 0.25 * ndr) * (0.55 + 0.45 * bottom_ao);
        col += vec3<f32>(0.72, 0.78, 0.88) * spec_rough * (0.45 + 0.55 * rim_top);
        col += sample_studio_bg(reflect(rd, n)) * fres * 0.45;
        col += metal_col * clamp(inlay * 0.55 + top_bevel * 0.22, 0.0, 0.75) * (0.35 + 0.65 * ndl + spec_rough * 1.4);

        let is_inner_face = inner_d <= 0.006 && abs(n.y) < 0.5 && dot(n.xz, -p.xz) > 0.0;
        if is_inner_face {
            let near_x = clamp(p.x, -hx + 0.08, hx - 0.08);
            let near_z = clamp(p.z, -hz + 0.08, hz - 0.08);
            let bounce = eval_floor_caustics(near_x, near_z, mat_id);
            let h_fade = exp(-max(p.y - fy, 0.0) * 5.2);
            let ripple = 0.65 + 0.35 * sin(p.y * 28.0 - studio_frame.eye_time.w * 2.5);
            col += bounce.rgb * h_fade * ripple * 0.22;
        }
        return col;
    }

    let box_d = length(max(abs(p.xz) - vec2<f32>(hx + rim_w, hz + rim_w), vec2<f32>(0.0)));
    let plinth_ao = smoothstep(0.0, 0.85, box_d);
    let fade = exp(-dot(p.xz, p.xz) * 0.022);
    let grain = value_noise(p.xz * 2.4);
    let fres = 0.02 + 0.24 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    var col = vec3<f32>(0.011, 0.013, 0.018) * (0.90 + 0.20 * grain) * (0.28 + 0.72 * plinth_ao);
    col += sample_studio_bg(reflect(rd, n)) * fres * 0.55 * plinth_ao;
    return mix(sample_studio_bg(rd), col, fade);
}

fn view_pos_at(coord: vec2<i32>, z_eye: f32) -> vec3<f32> {
    let uv = (vec2<f32>(coord) + 0.5) * studio_frame.viewport.zw;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ray_clip = studio_frame.inv_proj * vec4<f32>(ndc, 0.5, 1.0);
    let ray_view = ray_clip.xyz / max(ray_clip.w, 1.0e-5);
    let scale = z_eye / max(-ray_view.z, 1.0e-5);
    return ray_view * scale;
}

fn reconstruct_normal_view(coord: vec2<i32>, dims: vec2<i32>, z_c: f32, p_c: vec3<f32>) -> vec4<f32> {
    let c_l = clamp(coord - vec2<i32>(1, 0), vec2<i32>(0), dims - 1);
    let c_r = clamp(coord + vec2<i32>(1, 0), vec2<i32>(0), dims - 1);
    let c_u = clamp(coord - vec2<i32>(0, 1), vec2<i32>(0), dims - 1);
    let c_d = clamp(coord + vec2<i32>(0, 1), vec2<i32>(0), dims - 1);

    let z_l = textureLoad(studio_depth, c_l, 0).r;
    let z_r = textureLoad(studio_depth, c_r, 0).r;
    let z_u = textureLoad(studio_depth, c_u, 0).r;
    let z_d = textureLoad(studio_depth, c_d, 0).r;

    let edge_th = studio_frame.splat.x * 4.5;
    let dl = abs(z_l - z_c);
    let dr = abs(z_r - z_c);
    let du = abs(z_u - z_c);
    let dd = abs(z_d - z_c);
    let dl_ok = z_l > 0.0 && dl < edge_th;
    let dr_ok = z_r > 0.0 && dr < edge_th;
    let du_ok = z_u > 0.0 && du < edge_th;
    let dd_ok = z_d > 0.0 && dd < edge_th;

    let valid_count = f32(dl_ok) + f32(dr_ok) + f32(du_ok) + f32(dd_ok);
    let bg_count = f32(z_l <= 0.0) + f32(z_r <= 0.0) + f32(z_u <= 0.0) + f32(z_d <= 0.0);
    let outer_cov = select(1.0, smoothstep(4.0, 0.0, bg_count), bg_count > 0.5);
    let ridge_blend = smoothstep(0.5, 4.0, valid_count);

    var dpdx = vec3<f32>(1.0, 0.0, 0.0);
    if dl_ok && dr_ok {
        let curv_x = abs(0.5 * (z_l + z_r) - z_c);
        if curv_x < edge_th * 0.45 {
            dpdx = view_pos_at(c_r, z_r) - view_pos_at(c_l, z_l);
        } else if dr < dl {
            dpdx = view_pos_at(c_r, z_r) - p_c;
        } else {
            dpdx = p_c - view_pos_at(c_l, z_l);
        }
    } else if dr_ok {
        dpdx = view_pos_at(c_r, z_r) - p_c;
    } else if dl_ok {
        dpdx = p_c - view_pos_at(c_l, z_l);
    }

    var dpdy = vec3<f32>(0.0, -1.0, 0.0);
    if du_ok && dd_ok {
        let curv_y = abs(0.5 * (z_u + z_d) - z_c);
        if curv_y < edge_th * 0.45 {
            dpdy = view_pos_at(c_u, z_u) - view_pos_at(c_d, z_d);
        } else if du < dd {
            dpdy = view_pos_at(c_u, z_u) - p_c;
        } else {
            dpdy = p_c - view_pos_at(c_d, z_d);
        }
    } else if du_ok {
        dpdy = view_pos_at(c_u, z_u) - p_c;
    } else if dd_ok {
        dpdy = p_c - view_pos_at(c_d, z_d);
    }

    var n = cross(dpdx, dpdy);
    let len2 = dot(n, n);
    if len2 < 1.0e-12 {
        return vec4<f32>(0.0, 0.0, 1.0, outer_cov);
    }
    n = n * inverseSqrt(len2);
    if n.z < 0.0 {
        n = -n;
    }
    if valid_count < 3.5 {
        n = normalize(mix(normalize(n + vec3<f32>(0.0, 0.0, 0.35)), n, ridge_blend));
    }
    return vec4<f32>(n, outer_cov);
}

fn shade_chrome(p_w: vec3<f32>, n_w: vec3<f32>, v_w: vec3<f32>, thick: f32) -> vec3<f32> {
    let r_w = reflect(-v_w, n_w);
    var refl = sample_studio_hdri(r_w, 0.0);

    if r_w.y < -0.04 {
        let stage_fade = smoothstep(-0.04, -0.45, r_w.y);
        let dark_stage = vec3<f32>(0.06, 0.072, 0.092);
        refl = mix(refl, dark_stage, stage_fade * 0.55);
    }

    let ndv = clamp(dot(n_w, v_w), 0.0, 1.0);
    let f0 = vec3<f32>(0.94, 0.96, 0.99);
    let fres = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - ndv, 5.0);
    let floor_ao = smoothstep(0.0, 0.14, p_w.y - studio_frame.basin.z);
    let cool_edge = mix(vec3<f32>(1.0), vec3<f32>(0.88, 0.94, 1.06), pow(1.0 - ndv, 3.0));
    let depth_polish = 0.88 + 0.12 * smoothstep(0.0, 0.40, thick);
    return refl * fres * cool_edge * (0.76 + 0.24 * floor_ao) * depth_polish;
}

fn shade_water(p_w: vec3<f32>, n_w: vec3<f32>, v_w: vec3<f32>, thick: f32, foam: f32, speed: f32) -> vec3<f32> {
    let ndv = clamp(dot(n_w, v_w), 0.0, 1.0);
    let fres = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);

    let r_w = reflect(-v_w, n_w);
    let refl_env = sample_studio_hdri(r_w, 1.0);

    let t_dir = refract(-v_w, n_w, 1.0 / 1.333);
    var under = refl_env * 0.72;
    if dot(t_dir, t_dir) > 1.0e-4 {
        let refr_ray = normalize(t_dir);
        let refr_hit = trace_studio(p_w - n_w * 0.012, refr_ray);
        if refr_hit.kind != 0u {
            let rp = p_w + refr_ray * refr_hit.t;
            under = shade_studio_hit(refr_hit, rp, refr_ray, 1.0);
        } else {
            under = sample_studio_bg(refr_ray) * 0.85;
        }
    }

    let optical_d = clamp(thick * 0.66, 0.02, 3.2);
    let absorb = exp(-optical_d * vec3<f32>(1.85, 0.34, 0.20));
    let body_tint = vec3<f32>(0.010, 0.22, 0.30);
    let key_dir = normalize(vec3<f32>(-0.28, 0.88, 0.38));
    let back_scatter = pow(clamp(dot(v_w, -key_dir + n_w * 0.55), 0.0, 1.0), 3.0);
    let sss = vec3<f32>(0.04, 0.52, 0.62) * back_scatter * (1.0 - exp(-optical_d * 0.95)) * 0.55;

    var col = under * absorb + body_tint * (1.0 - absorb) * 0.38 + sss;
    col = mix(col, refl_env, fres);

    let lace = value_noise(p_w.xz * 22.0 + vec2<f32>(studio_frame.eye_time.w * 0.8, -studio_frame.eye_time.w * 0.6));
    let crest_sig = foam * 1.15 + smoothstep(1.15, 2.25, speed) * 0.36;
    let crest_foam = smoothstep(0.36, 0.86, crest_sig) * smoothstep(0.34, 0.78, lace);
    let foam_lit = vec3<f32>(0.92, 0.97, 1.0) * (0.58 + 0.48 * max(dot(n_w, key_dir), 0.0));
    col = mix(col, foam_lit, clamp(crest_foam * 0.68, 0.0, 0.72));
    return col;
}

fn shade_gold(p_w: vec3<f32>, n_w: vec3<f32>, v_w: vec3<f32>, thick: f32, speed: f32, energy: f32) -> vec3<f32> {
    let ndv = clamp(dot(n_w, v_w), 0.0, 1.0);
    let r_w = reflect(-v_w, n_w);
    let refl_env = sample_studio_hdri(r_w, 2.0);
    let f0 = vec3<f32>(1.0, 0.76, 0.22);
    let fres = f0 + (vec3<f32>(1.0, 0.92, 0.68) - f0) * pow(1.0 - ndv, 5.0);
    let floor_ao = smoothstep(0.0, 0.16, p_w.y - studio_frame.basin.z);
    let metallic = refl_env * fres * (0.78 + 0.22 * floor_ao) * 1.12;

    let swirl = fbm2(p_w.xz * 4.8 + vec2<f32>(p_w.y * 3.2, studio_frame.eye_time.w * 0.40));
    let seam = pow(clamp(1.0 - abs(swirl * 2.0 - 1.0), 0.0, 1.0), 5.0);
    let heat = clamp(
        seam * 0.72
            + smoothstep(0.85, 2.15, speed + energy * 0.45) * 0.42
            + exp(-thick * 2.2) * 0.18,
        0.0,
        1.0,
    );
    let magma = mix(vec3<f32>(0.85, 0.16, 0.01), vec3<f32>(2.4, 0.98, 0.12), heat);
    let emissive = magma * heat * 0.34;
    return metallic + emissive;
}

fn shade_plasma(p_w: vec3<f32>, n_w: vec3<f32>, v_w: vec3<f32>, thick: f32, speed: f32, energy: f32) -> vec3<f32> {
    let ndv = clamp(dot(n_w, v_w), 0.0, 1.0);
    let rim = pow(1.0 - ndv, 3.2);
    let r_w = reflect(-v_w, n_w);
    let refl = sample_studio_hdri(r_w, 3.0);

    let t = studio_frame.eye_time.w;
    let w1 = fbm2(p_w.xz * 3.8 + vec2<f32>(p_w.y * 2.4 - t * 0.85, t * 0.60));
    let w2 = value_noise(p_w.xz * 7.8 + vec2<f32>(w1 * 2.4 + t * 0.55, -p_w.y * 3.2));
    let core = vec3<f32>(0.007, 0.006, 0.028) * (0.50 + 0.50 * exp(-thick * 0.75));

    let cyan = vec3<f32>(0.04, 1.15, 1.95);
    let violet = vec3<f32>(1.28, 0.14, 2.05);
    let bio_hue = mix(cyan, violet, smoothstep(0.35, 0.75, w1));

    let vein = pow(clamp(1.0 - abs(w2 * 2.0 - 1.0), 0.0, 1.0), 8.5);
    let surge = 0.42 + smoothstep(0.55, 1.90, speed + energy * 0.65) * 1.15;
    let edge_cyan = vec3<f32>(0.05, 1.25, 2.15) * rim * 1.55;
    let thin_aura = bio_hue * exp(-thick * 2.4) * 0.24;
    let filaments = bio_hue * vein * surge * 1.65;

    let glass_spec = refl * (0.08 + 0.82 * pow(1.0 - ndv, 4.0));
    return core + glass_spec + edge_cyan + thin_aura + filaments;
}

@fragment
fn fs_studio(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(studio_depth));
    let coord = clamp(vec2<i32>(pos.xy), vec2<i32>(0), dims - 1);
    let uv = (vec2<f32>(coord) + 0.5) * studio_frame.viewport.zw;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let mat_id = studio_frame.style.x;

    let ray_clip = studio_frame.inv_proj * vec4<f32>(ndc, 0.5, 1.0);
    let ray_view = normalize(ray_clip.xyz / max(ray_clip.w, 1.0e-5));
    let ro = studio_frame.eye_time.xyz;
    let rd = normalize((studio_frame.inv_view * vec4<f32>(ray_view, 0.0)).xyz);

    let hit = trace_studio(ro, rd);
    var bg_z_eye = 1.0e9;
    if hit.kind != 0u {
        let hp = ro + rd * hit.t;
        let hv = (studio_frame.view * vec4<f32>(hp, 1.0)).xyz;
        bg_z_eye = -hv.z;
    }

    let z_fluid = textureLoad(studio_depth, coord, 0).r;
    var col = vec3<f32>(0.0);

    if z_fluid > 0.0 && z_fluid < bg_z_eye + 0.025 {
        let p_view = view_pos_at(coord, z_fluid);
        let n_cov = reconstruct_normal_view(coord, dims, z_fluid, p_view);
        let n_view = n_cov.xyz;
        let sil_cov = n_cov.w;
        let p_w = (studio_frame.inv_view * vec4<f32>(p_view, 1.0)).xyz;
        let n_w = normalize((studio_frame.inv_view * vec4<f32>(n_view, 0.0)).xyz);
        let v_w = normalize(ro - p_w);

        let thick_sample = textureSampleLevel(studio_thick, studio_sampler, uv, 0.0);
        let aux_sample = textureLoad(studio_aux, coord, 0);
        let w_norm = max(thick_sample.w, 0.01);
        let smooth_foam = clamp(thick_sample.y / w_norm, 0.0, 1.0);
        let smooth_speed = clamp(thick_sample.z / w_norm, 0.0, 2.5);
        let foam = mix(smooth_foam, aux_sample.y, 0.30);
        let speed = mix(smooth_speed, aux_sample.x, 0.30);
        let energy = aux_sample.z;
        let thick = thick_sample.x;

        if mat_id < 0.5 {
            col = shade_chrome(p_w, n_w, v_w, thick);
        } else if mat_id < 1.5 {
            col = shade_water(p_w, n_w, v_w, thick, foam, speed);
        } else if mat_id < 2.5 {
            col = shade_gold(p_w, n_w, v_w, thick, speed, energy);
        } else {
            col = shade_plasma(p_w, n_w, v_w, thick, speed, energy);
        }

        if sil_cov < 0.99 {
            var bg_edge = sample_studio_bg(rd);
            if hit.kind != 0u {
                let hp = ro + rd * hit.t;
                bg_edge = shade_studio_hit(hit, hp, rd, mat_id);
            }
            col = mix(bg_edge, col, 0.45 + 0.55 * sil_cov);
        }
    } else if hit.kind != 0u {
        let hp = ro + rd * hit.t;
        col = shade_studio_hit(hit, hp, rd, mat_id);
    } else {
        col = sample_studio_bg(rd);
    }

    if studio_frame.stir.w > 0.5 {
        let stir_c = studio_frame.stir.xyz;
        let to_s = stir_c - ro;
        let proj = dot(to_s, rd);
        if proj > 0.0 {
            let perp = length(to_s - rd * proj);
            let ring = exp(-pow((perp - 0.36) * 28.0, 2.0)) * 0.45 + exp(-pow(perp * 6.5, 2.0)) * 0.08;
            col += vec3<f32>(0.72, 0.90, 1.0) * ring;
        }
    }

    return vec4<f32>(max(col, vec3<f32>(0.0)), 1.0);
}
