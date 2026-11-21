import package::fluid::common::FrameUniform;

@group(1) @binding(0) var<uniform> studio_frame: FrameUniform;
@group(1) @binding(1) var studio_depth: texture_2d<f32>;
@group(1) @binding(2) var studio_thick: texture_2d<f32>;
@group(1) @binding(3) var studio_sampler: sampler;
@group(1) @binding(4) var<storage, read> studio_heightfield: array<vec4<f32>>;

const KEY_DIR: vec3<f32> = vec3<f32>(-0.280393, 0.881235, 0.380533);
const RIM_DIR: vec3<f32> = vec3<f32>(0.541845, 0.702392, -0.461572);
const K1_U: vec3<f32> = vec3<f32>(0.910866, 0.160152, 0.380361);
const K1_V: vec3<f32> = vec3<f32>(-0.200926, -0.421945, 0.884076);
const K2_U: vec3<f32> = vec3<f32>(0.419832, 0.119952, 0.899640);
const K2_V: vec3<f32> = vec3<f32>(-0.701968, 0.701968, 0.120337);
const K3_C: vec3<f32> = vec3<f32>(0.804355, 0.241307, 0.542940);
const K3_U: vec3<f32> = vec3<f32>(-0.559301, 0.0, 0.828964);
const K4_C: vec3<f32> = vec3<f32>(-0.824131, 0.261310, -0.502519);
const K4_U: vec3<f32> = vec3<f32>(0.521856, 0.0, -0.853034);
const UP_AXIS: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);

fn pow3(x: f32) -> f32 {
    return x * x * x;
}

fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

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
    let s1 = sin((az + 1.15) * 0.5);
    let s2 = sin((az - 2.05) * 0.5);
    let neg_flag_1 = exp(-s1 * s1 * 14.0);
    let neg_flag_2 = exp(-s2 * s2 * 16.0);
    let neg_fill = clamp(neg_flag_1 + neg_flag_2, 0.0, 0.92);
    base *= (0.45 + 0.72 * studio_bays) * (1.0 - 0.86 * neg_fill);

    let eq_d = (d.y + 0.07) * 9.5;
    let dark_equator = exp(-eq_d * eq_d);
    base *= (1.0 - 0.78 * dark_equator);

    let hz_d = (d.y - 0.06) * 13.0;
    let horizon_strip = exp(-hz_d * hz_d);
    let strip_panels = smoothstep(0.0, 0.42, cos(az * 2.0 - 0.5)) * (1.0 - neg_fill);
    base += vec3<f32>(1.15, 1.26, 1.45) * horizon_strip * strip_panels;

    let ground_bounce = vec3<f32>(0.045, 0.052, 0.068) * smoothstep(0.0, -0.55, d.y);
    base = mix(base, ground_bounce, smoothstep(-0.02, -0.22, d.y));

    let k1 = rect_light(d, KEY_DIR, K1_U, K1_V, vec2<f32>(0.72, 0.20), 0.09);
    let k2 = rect_light(d, RIM_DIR, K2_U, K2_V, vec2<f32>(0.56, 0.12), 0.055);
    let k3 = rect_light(d, K3_C, K3_U, UP_AXIS, vec2<f32>(0.36, 0.09), 0.06);
    let k4 = rect_light(d, K4_C, K4_U, UP_AXIS, vec2<f32>(0.44, 0.09), 0.065);

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
    let gd = (d.y - 0.14) * 5.0;
    let glow = exp(-gd * gd) * 0.020;
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
    let row0 = u32(iz) * stride + u32(ix);
    let row1 = row0 + stride;

    let v00 = studio_heightfield[row0];
    let v10 = studio_heightfield[row0 + 1u];
    let v01 = studio_heightfield[row1];
    let v11 = studio_heightfield[row1 + 1u];
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
    let mag = length(acc) * 0.33333334;
    let web = 1.0 - clamp(mag, 0.0, 1.0);
    return web * web * web * sqrt(max(web, 1.0e-6)) * 2.25;
}

fn eval_floor_caustics(wx: f32, wz: f32, mat_id: f32) -> vec4<f32> {
    let hc = sample_heightfield(wx, wz);
    if mat_id < 0.5 {
        let shadow = clamp(1.0 - smoothstep(1.2, 48.0, hc.y) * 0.62, 0.34, 1.0);
        return vec4<f32>(vec3<f32>(0.0), shadow);
    }

    let eps = 0.095;
    let h_xp = sample_heightfield(wx + eps, wz);
    let h_xm = sample_heightfield(wx - eps, wz);
    let h_zp = sample_heightfield(wx, wz + eps);
    let h_zm = sample_heightfield(wx, wz - eps);

    let col_mass = (hc.y * 2.0 + h_xp.y + h_xm.y + h_zp.y + h_zm.y) * 0.16666667;
    let shadow = clamp(1.0 - smoothstep(1.2, 48.0, col_mass) * 0.62, 0.34, 1.0);
    let t = studio_frame.eye_time.w;

    if mat_id < 1.5 {
        if studio_frame.style.y > 0.5 {
            return vec4<f32>(vec3<f32>(0.02, 0.14, 0.18) * smoothstep(2.0, 28.0, col_mass) * 0.35, shadow);
        }
        let coverage = smoothstep(0.6, 5.5, col_mass);
        let lap = h_xp.x + h_xm.x + h_zp.x + h_zm.x - 4.0 * hc.x;
        let grad = vec2<f32>(h_xp.x - h_xm.x, h_zp.x - h_zm.x) / (2.0 * eps);
        let focus = clamp(-lap * 0.68 + length(grad) * 0.17, 0.0, 2.5);
        let warp = vec2<f32>(wx, wz) + grad * 0.068;
        let cr = caustic_pattern(warp + vec2<f32>(0.018, 0.008), t);
        let cg = caustic_pattern(warp, t);
        let cb = caustic_pattern(warp - vec2<f32>(0.018, 0.010), t);
        let dyn = (0.52 + focus * 1.25) * (0.88 + 0.35 * clamp(hc.z, 0.0, 1.5)) * coverage;
        let rgb = vec3<f32>(cr, cg, cb) * vec3<f32>(0.84, 1.06, 1.18) * dyn * 1.32;
        return vec4<f32>(rgb, shadow);
    }

    if mat_id < 2.5 {
        let glow = smoothstep(1.0, 18.0, col_mass) * (0.35 + 0.45 * clamp(hc.z * 0.7, 0.0, 1.2));
        return vec4<f32>(vec3<f32>(1.15, 0.42, 0.05) * glow * 0.42, shadow);
    }

    let glow = smoothstep(0.8, 16.0, col_mass) * (0.42 + 0.45 * clamp(hc.z * 0.8, 0.0, 1.3));
    let shift = 0.5 + 0.5 * sin(wx * 1.5 + wz * 1.3 - t * 1.7);
    let tint = mix(vec3<f32>(0.05, 0.78, 1.20), vec3<f32>(0.72, 0.16, 1.18), shift);
    return vec4<f32>(tint * glow * 0.48, shadow);
}

struct Hit {
    t: f32,
    normal: vec3<f32>,
    kind: u32,
}

fn intersect_box(ro: vec3<f32>, inv: vec3<f32>, rd_sign: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec4<f32> {
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
        n.x = -rd_sign.x;
    } else if t_enter == tsmall.y {
        n.y = -rd_sign.y;
    } else {
        n.z = -rd_sign.z;
    }
    return vec4<f32>(t_enter, n);
}

fn trace_studio(ro: vec3<f32>, rd: vec3<f32>) -> Hit {
    var best = Hit(1.0e9, UP_AXIS, 0u);
    let hx = studio_frame.basin.x;
    let hz = studio_frame.basin.y;
    let fy = studio_frame.basin.z;
    let rim_h = 0.17;
    let rim_w = 0.14;
    let plinth_y = -0.24;

    if rd.y < -1.0e-5 {
        let t_floor = (plinth_y - ro.y) / rd.y;
        if t_floor > 0.0 && t_floor < best.t {
            best = Hit(t_floor, UP_AXIS, 1u);
        }
        let t_basin = (fy - ro.y) / rd.y;
        if t_basin > 0.0 && t_basin < best.t {
            let p = ro + rd * t_basin;
            if abs(p.x) <= hx + 0.02 && abs(p.z) <= hz + 0.02 {
                best = Hit(t_basin, UP_AXIS, 2u);
            }
        }
    }

    let inv = 1.0 / select(rd, vec3<f32>(1.0e-6), abs(rd) < vec3<f32>(1.0e-6));
    let rd_sign = sign(rd);
    let outer_min = vec3<f32>(-(hx + rim_w), plinth_y, -(hz + rim_w));
    let outer_max = vec3<f32>(hx + rim_w, fy + rim_h, hz + rim_w);
    let outer = intersect_box(ro, inv, rd_sign, outer_min, outer_max);
    if outer.x < 0.0 {
        return best;
    }

    let walls = array<vec3<f32>, 8>(
        outer_min,
        vec3<f32>(-hx, fy + rim_h, hz + rim_w),
        vec3<f32>(hx, plinth_y, -(hz + rim_w)),
        outer_max,
        vec3<f32>(-hx, plinth_y, -(hz + rim_w)),
        vec3<f32>(hx, fy + rim_h, -hz),
        vec3<f32>(-hx, plinth_y, hz),
        vec3<f32>(hx, fy + rim_h, hz + rim_w),
    );
    for (var i = 0; i < 4; i += 1) {
        let h = intersect_box(ro, inv, rd_sign, walls[i * 2], walls[i * 2 + 1]);
        if h.x > 0.0 && h.x < best.t {
            best = Hit(h.x, h.yzw, 3u);
        }
    }

    let base_hit = intersect_box(ro, inv, rd_sign, outer_min, vec3<f32>(hx + rim_w, fy, hz + rim_w));
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

    if hit.kind == 2u {
        let tile = abs(fract(p.xz * 1.45 + 0.5) - 0.5);
        let grout = 1.0 - smoothstep(0.478, 0.494, max(tile.x, tile.y));
        let grain = fbm2(p.xz * 3.6);
        let v_base = 1.0 - abs(value_noise(p.xz * 2.8 + vec2<f32>(grain * 1.3, -grain)) * 2.0 - 1.0);
        let v3 = pow3(v_base);
        let vein = v3 * v3;

        var albedo = vec3<f32>(0.022, 0.026, 0.033) * (0.86 + 0.28 * grain) + vec3<f32>(0.013, 0.015, 0.020) * vein;
        if mat_id > 0.5 && mat_id < 1.5 {
            albedo = vec3<f32>(0.044, 0.068, 0.084) * (0.84 + 0.30 * grain) + vec3<f32>(0.020, 0.032, 0.040) * vein;
        }
        albedo *= (0.72 + 0.28 * grout);

        let edge_dist = min(hx - abs(p.x), hz - abs(p.z));
        let wall_ao = smoothstep(0.0, 0.34, edge_dist);
        let caust = eval_floor_caustics(p.x, p.z, mat_id);
        let ndl = max(dot(n, KEY_DIR), 0.0);
        let ndr = max(dot(n, RIM_DIR), 0.0);

        let r_dir = reflect(rd, n);
        let spec_env = sample_studio_bg(r_dir);
        let fres = 0.025 + 0.28 * pow5(1.0 - max(dot(n, v), 0.0));

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

        let ndl = max(dot(n, KEY_DIR), 0.0);
        let ndr = max(dot(n, RIM_DIR), 0.0);
        let h_key = normalize(KEY_DIR + v);
        let h_rim = normalize(RIM_DIR + v);
        let spec_rough = pow(max(dot(n, h_key), 0.0), 42.0) * 0.28 + pow(max(dot(n, h_rim), 0.0), 32.0) * 0.16;
        let fres = 0.03 + 0.25 * pow5(1.0 - max(dot(n, v), 0.0));

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
    let fres = 0.02 + 0.24 * pow5(1.0 - max(dot(n, v), 0.0));
    var col = vec3<f32>(0.011, 0.013, 0.018) * (0.90 + 0.20 * grain) * (0.28 + 0.72 * plinth_ao);
    col += sample_studio_bg(reflect(rd, n)) * fres * 0.55 * plinth_ao;
    return mix(sample_studio_bg(rd), col, fade);
}

fn load_depth(c: vec2<i32>, dims: vec2<i32>) -> f32 {
    return textureLoad(studio_depth, clamp(c, vec2<i32>(0), dims - 1), 0).r;
}

fn edge_deriv(z_l: f32, z_c: f32, z_r: f32, edge_th: f32) -> f32 {
    let dl = abs(z_l - z_c);
    let dr = abs(z_r - z_c);
    let l_ok = z_l > 0.0 && dl < edge_th;
    let r_ok = z_r > 0.0 && dr < edge_th;
    if l_ok && r_ok {
        let curv = abs(0.5 * (z_l + z_r) - z_c);
        if curv < edge_th * 0.45 {
            return 0.5 * (z_r - z_l);
        }
        return select(z_c - z_l, z_r - z_c, dr < dl);
    }
    if r_ok {
        return z_r - z_c;
    }
    if l_ok {
        return z_c - z_l;
    }
    return 0.0;
}

fn normal_from_grad(ndc: vec2<f32>, dims: vec2<i32>, z_eye: f32, dzdx: f32, dzdy: f32) -> vec3<f32> {
    let rx = ndc.x * studio_frame.inv_proj[0][0];
    let ry = ndc.y * studio_frame.inv_proj[1][1];
    let dx_step = 2.0 * studio_frame.inv_proj[0][0] / f32(dims.x);
    let dy_step = -2.0 * studio_frame.inv_proj[1][1] / f32(dims.y);
    let dpdx = vec3<f32>(dx_step * z_eye + rx * dzdx, ry * dzdx, -dzdx);
    let dpdy = vec3<f32>(rx * dzdy, dy_step * z_eye + ry * dzdy, -dzdy);
    let n = cross(dpdy, dpdx);
    let len2 = dot(n, n);
    if len2 < 1.0e-12 {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    var nn = n * inverseSqrt(len2);
    if nn.z < 0.0 {
        nn = -nn;
    }
    return nn;
}

struct DepthNormal {
    z: f32,
    cov: f32,
    n_view: vec3<f32>,
}

fn sample_fluid_surface(pos_xy: vec2<f32>, uv: vec2<f32>, ndc: vec2<f32>, dims: vec2<i32>) -> DepthNormal {
    let edge_th = studio_frame.splat.x * 4.5;
    if abs(f32(dims.x) - studio_frame.viewport.x) < 1.5 {
        let coord = clamp(vec2<i32>(pos_xy), vec2<i32>(0), dims - 1);
        let z_c = load_depth(coord, dims);
        if z_c <= 0.0 {
            return DepthNormal(0.0, 0.0, vec3<f32>(0.0, 0.0, 1.0));
        }
        let z_l = load_depth(coord - vec2<i32>(1, 0), dims);
        let z_r = load_depth(coord + vec2<i32>(1, 0), dims);
        let z_u = load_depth(coord - vec2<i32>(0, 1), dims);
        let z_d = load_depth(coord + vec2<i32>(0, 1), dims);
        let dzdx = edge_deriv(z_l, z_c, z_r, edge_th);
        let dzdy = edge_deriv(z_u, z_c, z_d, edge_th);
        let bg_count = f32(z_l <= 0.0) + f32(z_r <= 0.0) + f32(z_u <= 0.0) + f32(z_d <= 0.0);
        let cov = select(1.0, 0.45 + 0.55 * smoothstep(4.0, 0.0, bg_count), bg_count > 0.5);
        let valid = f32(z_l > 0.0 && abs(z_l - z_c) < edge_th)
            + f32(z_r > 0.0 && abs(z_r - z_c) < edge_th)
            + f32(z_u > 0.0 && abs(z_u - z_c) < edge_th)
            + f32(z_d > 0.0 && abs(z_d - z_c) < edge_th);
        var n = normal_from_grad(ndc, dims, z_c, dzdx, dzdy);
        if valid < 3.5 {
            n = normalize(mix(normalize(n + vec3<f32>(0.0, 0.0, 0.35)), n, smoothstep(0.5, 4.0, valid)));
        }
        return DepthNormal(z_c, cov, n);
    }

    let p = uv * vec2<f32>(dims) - 0.5;
    let b = vec2<i32>(floor(p));
    let f = clamp(p - vec2<f32>(b), vec2<f32>(0.0), vec2<f32>(1.0));

    let z00 = load_depth(b, dims);
    let z10 = load_depth(b + vec2<i32>(1, 0), dims);
    let z01 = load_depth(b + vec2<i32>(0, 1), dims);
    let z11 = load_depth(b + vec2<i32>(1, 1), dims);

    let bw00 = (1.0 - f.x) * (1.0 - f.y);
    let bw10 = f.x * (1.0 - f.y);
    let bw01 = (1.0 - f.x) * f.y;
    let bw11 = f.x * f.y;

    let cov = bw00 * f32(z00 > 0.0) + bw10 * f32(z10 > 0.0) + bw01 * f32(z01 > 0.0) + bw11 * f32(z11 > 0.0);
    if cov < 0.22 {
        return DepthNormal(0.0, 0.0, vec3<f32>(0.0, 0.0, 1.0));
    }

    let z_row0 = select(z00, z10, f.x >= 0.5);
    let z_row1 = select(z01, z11, f.x >= 0.5);
    var z_ref = select(z_row0, z_row1, f.y >= 0.5);
    if z_ref <= 0.0 {
        var z_near = 1.0e9;
        if z00 > 0.0 { z_near = min(z_near, z00); }
        if z10 > 0.0 { z_near = min(z_near, z10); }
        if z01 > 0.0 { z_near = min(z_near, z01); }
        if z11 > 0.0 { z_near = min(z_near, z11); }
        z_ref = z_near;
    }

    let w00 = select(0.0, bw00, z00 > 0.0 && abs(z00 - z_ref) < edge_th);
    let w10 = select(0.0, bw10, z10 > 0.0 && abs(z10 - z_ref) < edge_th);
    let w01 = select(0.0, bw01, z01 > 0.0 && abs(z01 - z_ref) < edge_th);
    let w11 = select(0.0, bw11, z11 > 0.0 && abs(z11 - z_ref) < edge_th);
    let w_sum = max(w00 + w10 + w01 + w11, 1.0e-5);
    let z_eye = (z00 * w00 + z10 * w10 + z01 * w01 + z11 * w11) / w_sum;

    let zm0 = load_depth(b + vec2<i32>(-1, 0), dims);
    let z20 = load_depth(b + vec2<i32>(2, 0), dims);
    let zm1 = load_depth(b + vec2<i32>(-1, 1), dims);
    let z21 = load_depth(b + vec2<i32>(2, 1), dims);
    let z0m = load_depth(b + vec2<i32>(0, -1), dims);
    let z1m = load_depth(b + vec2<i32>(1, -1), dims);
    let z02 = load_depth(b + vec2<i32>(0, 2), dims);
    let z12 = load_depth(b + vec2<i32>(1, 2), dims);

    let g00 = vec2<f32>(edge_deriv(zm0, z00, z10, edge_th), edge_deriv(z0m, z00, z01, edge_th));
    let g10 = vec2<f32>(edge_deriv(z00, z10, z20, edge_th), edge_deriv(z1m, z10, z11, edge_th));
    let g01 = vec2<f32>(edge_deriv(zm1, z01, z11, edge_th), edge_deriv(z00, z01, z02, edge_th));
    let g11 = vec2<f32>(edge_deriv(z01, z11, z21, edge_th), edge_deriv(z10, z11, z12, edge_th));
    let grad = (g00 * w00 + g10 * w10 + g01 * w01 + g11 * w11) / w_sum;

    var n = normal_from_grad(ndc, dims, z_eye, grad.x, grad.y);
    if cov < 0.99 {
        n = normalize(mix(normalize(n + vec3<f32>(0.0, 0.0, 0.25)), n, smoothstep(0.22, 0.98, cov)));
    }
    return DepthNormal(z_eye, smoothstep(0.22, 0.85, cov), n);
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
    let omt = 1.0 - ndv;
    let f0 = vec3<f32>(0.94, 0.96, 0.99);
    let fres = f0 + (vec3<f32>(1.0) - f0) * pow5(omt);
    let floor_ao = smoothstep(0.0, 0.14, p_w.y - studio_frame.basin.z);
    let cool_edge = mix(vec3<f32>(1.0), vec3<f32>(0.88, 0.94, 1.06), pow3(omt));
    let depth_polish = 0.88 + 0.12 * smoothstep(0.0, 0.40, thick);
    return refl * fres * cool_edge * (0.76 + 0.24 * floor_ao) * depth_polish;
}

fn shade_water(p_w: vec3<f32>, n_w: vec3<f32>, v_w: vec3<f32>, thick: f32, foam: f32, speed: f32) -> vec3<f32> {
    let ndv = clamp(dot(n_w, v_w), 0.0, 1.0);
    let fres = 0.02 + 0.98 * pow5(1.0 - ndv);

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
    let back_scatter = pow3(clamp(dot(v_w, -KEY_DIR + n_w * 0.55), 0.0, 1.0));
    let sss = vec3<f32>(0.04, 0.52, 0.62) * back_scatter * (1.0 - exp(-optical_d * 0.95)) * 0.55;

    var col = under * absorb + body_tint * (1.0 - absorb) * 0.38 + sss;
    col = mix(col, refl_env, fres);

    let lace = value_noise(p_w.xz * 22.0 + vec2<f32>(studio_frame.eye_time.w * 0.8, -studio_frame.eye_time.w * 0.6));
    let crest_sig = foam * 1.15 + smoothstep(1.15, 2.25, speed) * 0.36;
    let crest_foam = smoothstep(0.36, 0.86, crest_sig) * smoothstep(0.34, 0.78, lace);
    let foam_lit = vec3<f32>(0.92, 0.97, 1.0) * (0.58 + 0.48 * max(dot(n_w, KEY_DIR), 0.0));
    col = mix(col, foam_lit, clamp(crest_foam * 0.68, 0.0, 0.72));
    return col;
}

fn shade_gold(p_w: vec3<f32>, n_w: vec3<f32>, v_w: vec3<f32>, thick: f32, speed: f32, energy: f32) -> vec3<f32> {
    let ndv = clamp(dot(n_w, v_w), 0.0, 1.0);
    let r_w = reflect(-v_w, n_w);
    let refl_env = sample_studio_hdri(r_w, 2.0);
    let f0 = vec3<f32>(1.0, 0.76, 0.22);
    let fres = f0 + (vec3<f32>(1.0, 0.92, 0.68) - f0) * pow5(1.0 - ndv);
    let floor_ao = smoothstep(0.0, 0.16, p_w.y - studio_frame.basin.z);
    let metallic = refl_env * fres * (0.78 + 0.22 * floor_ao) * 1.12;

    let swirl = fbm2(p_w.xz * 4.8 + vec2<f32>(p_w.y * 3.2, studio_frame.eye_time.w * 0.40));
    let seam = pow5(clamp(1.0 - abs(swirl * 2.0 - 1.0), 0.0, 1.0));
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
    let omt = 1.0 - ndv;
    let rim = pow3(omt) * sqrt(max(omt, 1.0e-6));
    let r_w = reflect(-v_w, n_w);
    let refl = sample_studio_hdri(r_w, 3.0);

    let t = studio_frame.eye_time.w;
    let w1 = fbm2(p_w.xz * 3.8 + vec2<f32>(p_w.y * 2.4 - t * 0.85, t * 0.60));
    let w2 = value_noise(p_w.xz * 7.8 + vec2<f32>(w1 * 2.4 + t * 0.55, -p_w.y * 3.2));
    let core = vec3<f32>(0.007, 0.006, 0.028) * (0.50 + 0.50 * exp(-thick * 0.75));

    let cyan = vec3<f32>(0.04, 1.15, 1.95);
    let violet = vec3<f32>(1.28, 0.14, 2.05);
    let bio_hue = mix(cyan, violet, smoothstep(0.35, 0.75, w1));

    let vb = clamp(1.0 - abs(w2 * 2.0 - 1.0), 0.0, 1.0);
    let v2 = vb * vb;
    let v4 = v2 * v2;
    let vein = v4 * v4 * sqrt(max(vb, 1.0e-6));
    let surge = 0.42 + smoothstep(0.55, 1.90, speed + energy * 0.65) * 1.15;
    let edge_cyan = vec3<f32>(0.05, 1.25, 2.15) * rim * 1.55;
    let thin_aura = bio_hue * exp(-thick * 2.4) * 0.24;
    let filaments = bio_hue * vein * surge * 1.65;

    let o2 = omt * omt;
    let glass_spec = refl * (0.08 + 0.82 * o2 * o2);
    return core + glass_spec + edge_cyan + thin_aura + filaments;
}

@fragment
fn fs_studio(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(studio_depth));
    let uv = pos.xy * studio_frame.viewport.zw;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let mat_id = studio_frame.style.x;

    let ray_view = normalize(vec3<f32>(
        ndc.x * studio_frame.inv_proj[0][0],
        ndc.y * studio_frame.inv_proj[1][1],
        -1.0,
    ));
    let ro = studio_frame.eye_time.xyz;
    let rd = normalize((studio_frame.inv_view * vec4<f32>(ray_view, 0.0)).xyz);

    let hit = trace_studio(ro, rd);
    var bg_z_eye = 1.0e9;
    if hit.kind != 0u {
        let hp = ro + rd * hit.t;
        let hv = (studio_frame.view * vec4<f32>(hp, 1.0)).xyz;
        bg_z_eye = -hv.z;
    }

    let surf = sample_fluid_surface(pos.xy, uv, ndc, dims);
    var col = vec3<f32>(0.0);

    if surf.z > 0.0 && surf.z < bg_z_eye + 0.025 {
        let p_view = vec3<f32>(
            ndc.x * studio_frame.inv_proj[0][0] * surf.z,
            ndc.y * studio_frame.inv_proj[1][1] * surf.z,
            -surf.z,
        );
        let p_w = (studio_frame.inv_view * vec4<f32>(p_view, 1.0)).xyz;
        let n_w = normalize((studio_frame.inv_view * vec4<f32>(surf.n_view, 0.0)).xyz);
        let v_w = normalize(ro - p_w);

        let thick_sample = textureSampleLevel(studio_thick, studio_sampler, uv, 0.0);
        let w_norm = max(thick_sample.w, 0.01);
        let meta_g = thick_sample.y / w_norm;
        let speed = clamp(thick_sample.z / w_norm, 0.0, 2.5);
        let thick = thick_sample.x;

        if mat_id < 0.5 {
            col = shade_chrome(p_w, n_w, v_w, thick);
        } else if mat_id < 1.5 {
            col = shade_water(p_w, n_w, v_w, thick, clamp(meta_g, 0.0, 1.0), speed);
        } else if mat_id < 2.5 {
            col = shade_gold(p_w, n_w, v_w, thick, speed, clamp(meta_g, 0.0, 1.5));
        } else {
            col = shade_plasma(p_w, n_w, v_w, thick, speed, clamp(meta_g, 0.0, 1.5));
        }

        if surf.cov < 0.99 {
            var bg_edge = sample_studio_bg(rd);
            if hit.kind != 0u {
                let hp = ro + rd * hit.t;
                bg_edge = shade_studio_hit(hit, hp, rd, mat_id);
            }
            col = mix(bg_edge, col, surf.cov);
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
            let r1 = (perp - 0.36) * 28.0;
            let r2 = perp * 6.5;
            let ring = exp(-r1 * r1) * 0.45 + exp(-r2 * r2) * 0.08;
            col += vec3<f32>(0.72, 0.90, 1.0) * ring;
        }
    }

    return vec4<f32>(max(col, vec3<f32>(0.0)), 1.0);
}
