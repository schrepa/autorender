import package::generated::frame::FrameUniforms;
import package::nanite::frame::nanite_frame_uniforms;
import package::nanite::frame::nanite_frame_instances;
import package::nanite::frame::nanite_frame_clusters;
import package::nanite::frame::nanite_frame_page_heap;
import package::nanite::frame::nanite_frame_vis_clusters;
import package::nanite::frame::nanite_frame_sw_depth;
import package::nanite::frame::nanite_frame_sw_vis;
import package::nanite::frame::nanite_frame_shadow_depth;
import package::nanite::common::PAGE_WORDS;
import package::nanite::common::SHADOW_C0_DIM;
import package::nanite::common::SHADOW_C1_DIM;
import package::nanite::common::quat_rotate;
import package::nanite::common::inst_to_world;
import package::nanite::common::decode_pos;
import package::nanite::common::decode_color_ao;
import package::nanite::common::decode_octahedral;
import package::nanite::common::decode_uv;
import package::nanite::common::cluster_vert_count;
import package::nanite::common::cluster_lod;
import package::nanite::common::cluster_mat;

@group(2) @binding(0) var vis_hw_tex: texture_2d<u32>;
@group(2) @binding(1) var hw_depth_tex: texture_depth_2d;
@group(2) @binding(2) var<uniform> frame_params: FrameUniforms;

const PI: f32 = 3.14159265;

fn hash_u32(v_in: u32) -> u32 {
    var v = v_in;
    v = (v ^ 61u) ^ (v >> 16u);
    v = v + (v << 3u);
    v = v ^ (v >> 4u);
    v = v * 0x27d4eb2du;
    v = v ^ (v >> 15u);
    return v;
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = fract(vec3<f32>(h, h + 0.6666667, h + 0.3333333)) * 6.0;
    let rgb = clamp(abs(k - 3.0) - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
    return v * mix(vec3<f32>(1.0), rgb, s);
}

fn hash_color(id: u32) -> vec3<f32> {
    let h = hash_u32(id);
    let hue = f32(h & 0xFFFFu) * (1.0 / 65536.0);
    let sat = 0.62 + 0.28 * f32((h >> 16u) & 0xFFu) * (1.0 / 255.0);
    let val = 0.68 + 0.28 * f32((h >> 24u) & 0xFFu) * (1.0 / 255.0);
    return hsv_to_rgb(hue, sat, val);
}

fn lod_heat_color(lod: u32) -> vec3<f32> {
    let t = clamp(f32(lod) * 0.14, 0.0, 1.0);
    let c0 = vec3<f32>(0.06, 0.34, 0.92);
    let c1 = vec3<f32>(0.05, 0.78, 0.62);
    let c2 = vec3<f32>(0.96, 0.78, 0.12);
    let c3 = vec3<f32>(0.94, 0.18, 0.24);
    if t < 0.33 {
        return mix(c0, c1, t * 3.0);
    } else if t < 0.66 {
        return mix(c1, c2, (t - 0.33) * 3.0);
    }
    return mix(c2, c3, (t - 0.66) * 3.0);
}

fn hash21(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

fn value_noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn eval_sky(dir: vec3<f32>, include_sun: bool) -> vec3<f32> {
    let sun_dir = frame_params.sun_dir_illum.xyz;
    let up_t = pow(clamp(dir.y, 0.0, 1.0), 0.32);
    let sun_az = clamp(dot(normalize(vec2<f32>(dir.x, dir.z) + vec2<f32>(1.0e-5, 0.0)), normalize(vec2<f32>(sun_dir.x, sun_dir.z) + vec2<f32>(1.0e-5, 0.0))) * 0.5 + 0.5, 0.0, 1.0);
    let horiz = mix(frame_params.sky_horiz_opp.rgb, frame_params.sky_horiz_sun.rgb, pow(sun_az, 1.35));
    var col = mix(horiz, frame_params.sky_zenith.rgb, up_t);

    if dir.y < 0.0 {
        let ground_fade = clamp(-dir.y * 4.5, 0.0, 1.0);
        col = mix(horiz, vec3<f32>(0.36, 0.25, 0.16), ground_fade);
    }

    let cos_theta = dot(dir, sun_dir);
    let g = frame_params.fog_color.w;
    let g2 = g * g;
    let denom = max(1.0 + g2 - 2.0 * g * cos_theta, 0.02);
    let mie = (1.0 - g2) / (4.0 * PI * pow(denom, 1.5));
    let mie_tint = vec3<f32>(1.0, 0.78, 0.52);
    col += mie_tint * (min(mie, 2.0) * 0.26 * (1.0 - up_t * 0.62));

    if include_sun {
        let corona = pow(clamp((cos_theta - 0.994) / 0.006, 0.0, 1.0), 3.0);
        let soft_glow = pow(clamp((cos_theta - 0.96) / 0.04, 0.0, 1.0), 4.0);
        let disk = smoothstep(0.999972, 0.999990, cos_theta);
        col += frame_params.sun_color_shaft.rgb * (soft_glow * 0.55 + corona * 2.5 + disk * 28.0);
    }
    return col;
}

fn shadow_texel(tx: i32, ty: i32, cmp_z: f32, dim_u: u32, dim_i: i32, base_off: u32) -> f32 {
    let cx = clamp(tx, 0, dim_i - 1);
    let cy = clamp(ty, 0, dim_i - 1);
    let bits = nanite_frame_shadow_depth[base_off + u32(cy) * dim_u + u32(cx)];
    return select(0.0, 1.0, bits == 0u || cmp_z >= bitcast<f32>(bits));
}

fn sample_shadow_bilinear(sx: f32, sy: f32, cmp_z: f32, dim_u: u32, dim_i: i32, base_off: u32) -> f32 {
    let gx = sx - 0.5;
    let gy = sy - 0.5;
    let x0 = i32(floor(gx));
    let y0 = i32(floor(gy));
    let fx = gx - f32(x0);
    let fy = gy - f32(y0);
    let s00 = shadow_texel(x0, y0, cmp_z, dim_u, dim_i, base_off);
    let s10 = shadow_texel(x0 + 1, y0, cmp_z, dim_u, dim_i, base_off);
    let s01 = shadow_texel(x0, y0 + 1, cmp_z, dim_u, dim_i, base_off);
    let s11 = shadow_texel(x0 + 1, y0 + 1, cmp_z, dim_u, dim_i, base_off);
    return mix(mix(s00, s10, fx), mix(s01, s11, fx), fy);
}

fn sample_cascade_pcf(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    vp: mat4x4<f32>,
    dim_u: u32,
    base_off: u32,
    normal_bias: f32,
    depth_bias: f32,
    rot_angle: f32,
    radius_px: f32,
) -> vec2<f32> {
    let p = world_pos + n * normal_bias;
    let clip = vp * vec4<f32>(p, 1.0);
    let margin = max(abs(clip.x), abs(clip.y));
    if margin >= 0.98 || clip.z <= 0.001 || clip.z >= 0.999 {
        return vec2<f32>(1.0, 0.0);
    }
    let dim_f = f32(dim_u);
    let dim_i = i32(dim_u);
    let sx = (clip.x * 0.5 + 0.5) * dim_f;
    let sy = (0.5 - clip.y * 0.5) * dim_f;
    let ca = cos(rot_angle);
    let sa = sin(rot_angle);
    let cmp_z = clip.z + depth_bias;
    var lit = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        let r = sqrt((f32(i) + 0.5) * 0.25) * radius_px;
        let a = f32(i) * 2.3999632;
        let ox = cos(a) * r;
        let oy = sin(a) * r;
        let rx = ox * ca - oy * sa;
        let ry = ox * sa + oy * ca;
        lit += sample_shadow_bilinear(sx + rx, sy + ry, cmp_z, dim_u, dim_i, base_off);
    }
    let coverage = smoothstep(0.98, 0.78, margin);
    return vec2<f32>(lit * 0.25, coverage);
}

fn eval_shadow(world_pos: vec3<f32>, n: vec3<f32>, frag_xy: vec2<f32>) -> f32 {
    let ign = fract(52.9829189 * fract(dot(frag_xy, vec2<f32>(0.06711056, 0.00583715))));
    let angle = ign * 6.2831853;
    let ndl = max(dot(n, frame_params.sun_dir_illum.xyz), 0.0);
    let slope = clamp(1.0 - ndl, 0.0, 1.0);
    let c0 = sample_cascade_pcf(
        world_pos,
        n,
        nanite_frame_uniforms.shadow_vp0,
        SHADOW_C0_DIM,
        0u,
        0.06 + slope * 0.10,
        0.00030 + slope * 0.00045,
        angle,
        1.35,
    );
    if c0.y >= 0.99 {
        return c0.x;
    }
    let c1 = sample_cascade_pcf(
        world_pos,
        n,
        nanite_frame_uniforms.shadow_vp1,
        SHADOW_C1_DIM,
        SHADOW_C0_DIM * SHADOW_C0_DIM,
        0.14 + slope * 0.22,
        0.00035 + slope * 0.00055,
        angle,
        1.25,
    );
    let far_val = mix(1.0, c1.x, c1.y);
    return mix(far_val, c0.x, c0.y);
}

fn distribution_ggx(ndh: f32, rough: f32) -> f32 {
    let a = rough * rough;
    let a2 = a * a;
    let d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1.0e-5);
}

fn geometry_smith(ndv: f32, ndl: f32, rough: f32) -> f32 {
    let r = rough + 1.0;
    let k = (r * r) * 0.125;
    let gv = ndv / (ndv * (1.0 - k) + k);
    let gl = ndl / (ndl * (1.0 - k) + k);
    return gv * gl;
}

fn fresnel_schlick(cos_t: f32, f0: vec3<f32>) -> vec3<f32> {
    let f = pow(clamp(1.0 - cos_t, 0.0, 1.0), 5.0);
    return f0 + (vec3<f32>(1.0) - f0) * f;
}

fn apply_aerial_fog(
    color: vec3<f32>,
    world_pos: vec3<f32>,
    cam_pos: vec3<f32>,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    let dist = length(world_pos - cam_pos);
    let density = frame_params.fog_params.x;
    let falloff = frame_params.fog_params.y;
    let base_y = frame_params.fog_params.z;
    let avg_h = max(0.5 * (cam_pos.y + world_pos.y) - base_y, 0.0);
    let optical = density * dist * exp(-falloff * avg_h);
    let fog_amt = 1.0 - exp(-min(optical, 5.5));
    let sun_dot = clamp(dot(view_dir, frame_params.sun_dir_illum.xyz), 0.0, 1.0);
    let mie_glow = pow(sun_dot, 5.0) * 0.35 + pow(sun_dot, 16.0) * 0.45;
    let horiz_mix = mix(frame_params.sky_horiz_opp.rgb, frame_params.sky_horiz_sun.rgb, sun_dot);
    let sky_haze = mix(horiz_mix, frame_params.sky_zenith.rgb, clamp(view_dir.y * 1.4, 0.0, 0.7));
    let fog_col = mix(frame_params.fog_color.rgb, sky_haze, 0.42) + vec3<f32>(1.0, 0.78, 0.55) * mie_glow;
    return mix(color, fog_col, clamp(fog_amt, 0.0, 0.92));
}

fn shade_pbr(
    world_pos: vec3<f32>,
    n_in: vec3<f32>,
    base_color: vec3<f32>,
    ao: f32,
    mat_id: u32,
    uv: vec2<f32>,
    uv_filter: f32,
    normal_var: f32,
    frag_xy: vec2<f32>,
) -> vec3<f32> {
    let cam_pos = nanite_frame_uniforms.render_pos_time.xyz;
    let to_cam = cam_pos - world_pos;
    let v_len = max(length(to_cam), 1.0e-4);
    let v = to_cam / v_len;
    let n = select(n_in, -n_in, dot(n_in, v) < -0.15);
    let l = frame_params.sun_dir_illum.xyz;
    let sun_rad = frame_params.sun_color_shaft.rgb * frame_params.sun_dir_illum.w;

    let grain_fade = 1.0 - smoothstep(0.04, 0.25, uv_filter);
    let micro = (value_noise2(uv * 7.5) - 0.5) * 0.08 * grain_fade;

    var albedo = base_color;
    var rough = 0.65;
    var metallic = 0.0;
    var wrap_amt = 0.0;
    var wrap_tint = vec3<f32>(0.0);
    var backlit_amt = 0.0;

    if mat_id == 0u {
        albedo = clamp(base_color * (1.0 + micro * 0.7), vec3<f32>(0.02), vec3<f32>(0.98));
        rough = mix(0.56, 0.42, smoothstep(0.55, 0.92, n.y));
        wrap_amt = 0.36;
        wrap_tint = vec3<f32>(1.00, 0.80, 0.58);
    } else if mat_id == 1u {
        let dust = smoothstep(0.58, 0.92, n.y) * 0.45;
        albedo = mix(base_color, vec3<f32>(0.30, 0.20, 0.12), dust) * (1.0 + micro);
        rough = 0.86;
    } else if mat_id == 2u {
        let verd = smoothstep(-0.006, 0.042, base_color.g - base_color.r);
        albedo = clamp(base_color * (1.0 + micro * 0.5), vec3<f32>(0.02), vec3<f32>(0.98));
        rough = mix(0.28, 0.68, verd);
        metallic = mix(1.0, 0.2, verd);
    } else if mat_id == 3u {
        albedo = clamp(base_color * (1.0 + micro), vec3<f32>(0.02), vec3<f32>(0.98));
        rough = 0.80;
    } else {
        albedo = clamp(base_color * (1.0 + micro * 0.5), vec3<f32>(0.01), vec3<f32>(0.90));
        rough = 0.62;
        backlit_amt = 0.35;
        wrap_tint = vec3<f32>(0.35, 0.55, 0.12);
    }

    rough = sqrt(clamp(rough * rough + normal_var * 0.50, 0.04, 0.96));

    let ndl = max(dot(n, l), 0.0);
    var sh = 0.0;
    if ndl > 0.001 || wrap_amt > 0.0 || backlit_amt > 0.0 || mat_id == 2u {
        sh = eval_shadow(world_pos, n, frag_xy);
    }
    let ndv = max(dot(n, v), 0.001);
    let h_raw = l + v;
    let h_l2 = dot(h_raw, h_raw);
    let h = select(n, h_raw * inverseSqrt(h_l2), h_l2 > 1.0e-6);
    let ndh = max(dot(n, h), 0.0);
    let vdh = max(dot(v, h), 0.0);

    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let d_term = distribution_ggx(ndh, rough);
    let g_term = geometry_smith(ndv, ndl, rough);
    let f_term = fresnel_schlick(vdh, f0);
    let spec = (d_term * g_term * f_term) / max(4.0 * ndv * ndl, 1.0e-4);
    let kd = (vec3<f32>(1.0) - f_term) * (1.0 - metallic);
    let diff = kd * albedo * (1.0 / PI);

    var direct = (diff + spec) * sun_rad * ndl * sh;

    if wrap_amt > 0.0 {
        let wrap_ndl = clamp((dot(n, l) + wrap_amt) / (1.0 + wrap_amt), 0.0, 1.0);
        let grazing = pow(clamp(1.0 - ndv, 0.0, 1.0), 2.15);
        let back_sun = pow(clamp(dot(-v, l), 0.0, 1.0), 2.2);
        let sss = max(wrap_ndl - ndl, 0.0) * 0.62 + grazing * (0.36 + 0.78 * back_sun) * (0.40 + 0.60 * sh);
        direct += albedo * wrap_tint * sun_rad * (sss * wrap_amt * (1.0 / PI));
    }
    if backlit_amt > 0.0 {
        let trans = pow(clamp(dot(-v, l), 0.0, 1.0), 3.0) * (0.35 + 0.65 * sh);
        let wrap_leaf = clamp((dot(n, l) + 0.35) / 1.35, 0.0, 1.0) * 0.3 * sh;
        direct += wrap_tint * sun_rad * ((trans + wrap_leaf) * backlit_amt * (1.0 / PI));
    }
    if mat_id == 2u {
        let rim = pow(clamp(1.0 - ndv, 0.0, 1.0), 2.6);
        let back_s = pow(clamp(dot(-v, l), 0.0, 1.0), 2.0);
        direct += albedo * frame_params.sun_color_shaft.rgb * (rim * (0.45 + 1.15 * back_s) * metallic * (0.4 + 0.6 * sh));
    }

    let sky_avg = mix(frame_params.sky_horiz_sun.rgb, frame_params.sky_horiz_opp.rgb, 0.5);
    let sky_hemi = mix(sky_avg, frame_params.sky_zenith.rgb, clamp(n.y * 0.5 + 0.5, 0.0, 1.0));
    let ground_bounce = vec3<f32>(0.56, 0.39, 0.25) * 0.48;
    let opp_bounce = vec3<f32>(0.90, 0.66, 0.42) * (clamp(dot(n, -l), 0.0, 1.0) * 0.26 * (1.0 - metallic * 0.65));
    let hemi = mix(ground_bounce, sky_hemi, n.y * 0.5 + 0.5) + opp_bounce;
    let ao_spec = clamp(pow(ndv + ao, exp2(-16.0 * rough - 1.0)) - 1.0 + ao, 0.0, 1.0);
    var env_col = sky_hemi;
    if metallic > 0.05 || rough < 0.45 {
        let refl = reflect(-v, n);
        env_col = eval_sky(refl, false);
    }
    let f_env = fresnel_schlick(ndv, f0);
    let env_spec = env_col * f_env * (1.0 - rough * 0.68) * ao_spec * (0.50 + 0.85 * metallic);
    let indirect = hemi * albedo * (1.0 - metallic * 0.76) * (ao * 0.60) + env_spec;

    let lit = direct + indirect;
    return apply_aerial_fog(lit, world_pos, cam_pos, -v);
}

fn ground_normal(nid: u32) -> vec3<f32> {
    if nid == 1u { return vec3<f32>(0.0, 0.0, 1.0); }
    if nid == 2u { return vec3<f32>(0.0, 0.0, -1.0); }
    if nid == 3u { return vec3<f32>(1.0, 0.0, 0.0); }
    if nid == 4u { return vec3<f32>(-1.0, 0.0, 0.0); }
    return vec3<f32>(0.0, 1.0, 0.0);
}

fn paver_grid(p: vec2<f32>, tile: f32, joint: f32, fp: f32, stagger: bool) -> vec2<f32> {
    var q = p / tile;
    let row = floor(q.y);
    if stagger {
        q.x += (row - floor(row * 0.5) * 2.0) * 0.5;
    }
    let cell = floor(q);
    let f = fract(q);
    let d_edge = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y)) * tile;
    let aa = max(fp * 1.35, 0.002);
    let joint_mask = 1.0 - smoothstep(joint * 0.5 - aa, joint * 0.5 + aa, d_edge);
    let jitter = hash21(cell);
    return vec2<f32>(joint_mask * (1.0 - smoothstep(tile * 0.35, tile * 0.9, fp)), jitter);
}

fn shade_ground(
    vis_id: u32,
    ndc_p: vec2<f32>,
    depth: f32,
    view_mode: u32,
    frag_xy: vec2<f32>,
) -> vec3<f32> {
    let inv_vp = nanite_frame_uniforms.inv_render_vp;
    let h = inv_vp * vec4<f32>(ndc_p, depth, 1.0);
    let world_pos = h.xyz / max(abs(h.w), 1.0e-5) * sign(h.w);
    let mat_id = (vis_id >> 8u) & 0xFu;
    let style = (vis_id >> 4u) & 0xFu;
    let n = ground_normal(vis_id & 0x7u);

    let cam_pos = nanite_frame_uniforms.render_pos_time.xyz;
    let dist = length(world_pos - cam_pos);
    if view_mode != 0u {
        let l = frame_params.sun_dir_illum.xyz;
        let ndl = 0.45 + 0.55 * clamp(dot(n, l) * 0.5 + 0.5, 0.0, 1.0);
        let base_g = select(vec3<f32>(0.24, 0.23, 0.22), vec3<f32>(0.15, 0.17, 0.21), view_mode == 4u) * ndl;
        if style == 0u {
            return mix(base_g, vec3<f32>(0.05, 0.055, 0.07), smoothstep(520.0, 2200.0, dist));
        }
        return base_g;
    }

    let fp = max(dist / max(nanite_frame_uniforms.shadow_info.z, 1.0), 0.002);
    var albedo = vec3<f32>(0.784, 0.701, 0.565);
    var ao = 1.0;

    if style == 0u {
        let sand_base = vec3<f32>(0.578, 0.332, 0.156) * 0.85;
        let macro_n = value_noise2(world_pos.xz * 0.018);
        let ripple = sin(world_pos.x * 0.22 + world_pos.z * 0.58 + macro_n * 3.0) * 0.045;
        albedo = sand_base * (0.88 + 0.24 * macro_n + ripple * (1.0 - smoothstep(0.18, 0.75, fp)));
    } else if style == 1u {
        let pg = paver_grid(world_pos.xz, 1.5, 0.014, fp, true);
        let vein = value_noise2(world_pos.xz * 0.35 + vec2<f32>(pg.y * 13.0, pg.y * 7.0));
        let border = smoothstep(0.35, 0.05, abs(abs(world_pos.x) - 10.6));
        let slab = mix(vec3<f32>(0.784, 0.701, 0.565), vec3<f32>(0.68, 0.58, 0.44), border * 0.65);
        albedo = mix(slab * (0.91 + 0.14 * pg.y + (vein - 0.5) * 0.08), vec3<f32>(0.26, 0.16, 0.10), pg.x);
        ao = 1.0 - pg.x * 0.45;
    } else if style == 2u {
        let pg = paver_grid(world_pos.xz, 1.8, 0.018, fp, true);
        let sand = vec3<f32>(0.578, 0.332, 0.156);
        albedo = mix(sand * (0.88 + 0.20 * pg.y), vec3<f32>(0.22, 0.13, 0.08), pg.x);
        ao = 1.0 - pg.x * 0.4;
    } else if style == 3u {
        let uv_t = select(world_pos.xz, vec2<f32>(world_pos.x + world_pos.z, world_pos.y), n.y < 0.5);
        let pg = paver_grid(uv_t, 2.0, 0.016, fp, true);
        albedo = mix(vec3<f32>(0.76, 0.67, 0.53) * (0.92 + 0.12 * pg.y), vec3<f32>(0.28, 0.17, 0.10), pg.x);
        ao = 1.0 - pg.x * 0.4;
    } else if style == 4u {
        let uv_t = select(world_pos.xz, vec2<f32>(world_pos.x + world_pos.z, world_pos.y), n.y < 0.5);
        let pg = paver_grid(uv_t, 1.8, 0.018, fp, true);
        albedo = mix(vec3<f32>(0.578, 0.332, 0.156) * (0.90 + 0.16 * pg.y), vec3<f32>(0.22, 0.12, 0.07), pg.x);
        ao = 1.0 - pg.x * 0.4;
    } else if style == 5u {
        let rel = world_pos.xz - vec2<f32>(0.0, -205.0);
        let rad = length(rel);
        let ang = atan2(rel.y, rel.x);
        let ring_f = fract(rad / 3.0);
        let ring_d = min(ring_f, 1.0 - ring_f) * 3.0;
        let aa = max(fp * 1.4, 0.003);
        let ring_j = 1.0 - smoothstep(0.012 - aa, 0.012 + aa, ring_d);
        let spoke_step = (2.0 * PI) / 24.0;
        let spoke_f = fract((ang + PI) / spoke_step);
        let spoke_d = min(spoke_f, 1.0 - spoke_f) * spoke_step * max(rad, 1.0);
        let spoke_j = (1.0 - smoothstep(0.012 - aa, 0.012 + aa, spoke_d)) * smoothstep(2.0, 4.0, rad);
        let j = max(ring_j, spoke_j) * (1.0 - smoothstep(0.7, 2.2, fp));
        let ring_id = floor(rad / 3.0);
        let spoke_id = floor((ang + PI) / spoke_step);
        let tint = 0.90 + 0.15 * hash21(vec2<f32>(ring_id, spoke_id));
        albedo = mix(vec3<f32>(0.784, 0.701, 0.565) * tint, vec3<f32>(0.26, 0.16, 0.10), j);
        ao = 1.0 - j * 0.45;
    } else {
        let uv_s = select(world_pos.xz, vec2<f32>(world_pos.x, world_pos.y), n.y < 0.5);
        let pg = paver_grid(uv_s, 1.6, 0.014, fp, false);
        albedo = mix(vec3<f32>(0.77, 0.68, 0.54) * (0.93 + 0.10 * pg.y), vec3<f32>(0.28, 0.18, 0.11), pg.x);
        ao = select(1.0, 0.82, n.y < 0.5) * (1.0 - pg.x * 0.35);
    }

    let shaded = shade_pbr(world_pos, n, albedo, ao, mat_id, world_pos.xz, fp, 0.0, frag_xy);
    if style == 0u {
        let to_p = world_pos - cam_pos;
        let vd = to_p / max(dist, 1.0e-4);
        let horiz_sky = eval_sky(vec3<f32>(vd.x, max(vd.y, 0.012), vd.z), false);
        let h_blend = smoothstep(500.0, 2200.0, dist);
        return mix(shaded, horiz_sky, h_blend);
    }
    return shaded;
}

@fragment
fn fs_resolve(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let vp = nanite_frame_uniforms.viewport_lod.xy;
    let size = max(vec2<u32>(vp), vec2<u32>(1u));
    let px = min(vec2<u32>(pos.xy), size - vec2<u32>(1u));
    let px_i = vec2<i32>(px);
    let idx = px.y * size.x + px.x;
    let uv = pos.xy / max(vp, vec2<f32>(1.0));
    let ndc_p = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let view_mode = nanite_frame_uniforms.flags.z;

    let hw_d = textureLoad(hw_depth_tex, px_i, 0);
    let hw_v = textureLoad(vis_hw_tex, px_i, 0).r;
    let sw_d_bits = nanite_frame_sw_depth[idx];
    let sw_d = bitcast<f32>(sw_d_bits);
    let sw_v = nanite_frame_sw_vis[idx];
    let use_sw = (sw_v != 0u) && (sw_d > hw_d || hw_v == 0u);
    let vis_id = select(hw_v, sw_v, use_sw);
    let win_depth = select(hw_d, sw_d, use_sw);

    if vis_id == 0u {
        if view_mode != 0u {
            return vec4<f32>(0.05, 0.055, 0.07, 0.0);
        }
        let inv_vp = nanite_frame_uniforms.inv_render_vp;
        let far_h = inv_vp * vec4<f32>(ndc_p, 0.0001, 1.0);
        let far_w = far_h.xyz / max(abs(far_h.w), 1.0e-5) * sign(far_h.w);
        let cam_pos = nanite_frame_uniforms.render_pos_time.xyz;
        let ray_d = normalize(far_w - cam_pos);
        return vec4<f32>(eval_sky(ray_d, true), 0.0);
    }

    if (vis_id & 0x80000000u) != 0u {
        let gcol = shade_ground(vis_id, ndc_p, win_depth, view_mode, pos.xy);
        return vec4<f32>(gcol, 1.0);
    }

    let vis_slot = (vis_id >> 6u) - 1u;
    let tri_id = vis_id & 63u;
    let vc = nanite_frame_vis_clusters[vis_slot];
    let c = nanite_frame_clusters[vc.cluster_id];
    let inst = nanite_frame_instances[vc.instance_id];
    let vert_count = cluster_vert_count(c);
    let page_base = vc.heap_slot * PAGE_WORDS + c.cone_and_meta.w;
    let tri_word = nanite_frame_page_heap[page_base + vert_count * 4u + tri_id];
    let i0 = tri_word & 0xFFu;
    let i1 = (tri_word >> 8u) & 0xFFu;
    let i2 = (tri_word >> 16u) & 0xFFu;

    let b0 = page_base + i0 * 4u;
    let b1 = page_base + i1 * 4u;
    let b2 = page_base + i2 * 4u;

    let w0_0 = nanite_frame_page_heap[b0];
    let w1_0 = nanite_frame_page_heap[b0 + 1u];
    let w0_1 = nanite_frame_page_heap[b1];
    let w1_1 = nanite_frame_page_heap[b1 + 1u];
    let w0_2 = nanite_frame_page_heap[b2];
    let w1_2 = nanite_frame_page_heap[b2 + 1u];

    let p0 = inst_to_world(inst, decode_pos(w0_0, w1_0, c.aabb_min.xyz, c.aabb_extent.xyz));
    let p1 = inst_to_world(inst, decode_pos(w0_1, w1_1, c.aabb_min.xyz, c.aabb_extent.xyz));
    let p2 = inst_to_world(inst, decode_pos(w0_2, w1_2, c.aabb_min.xyz, c.aabb_extent.xyz));

    let rvp = nanite_frame_uniforms.render_view_proj;
    let c0 = rvp * vec4<f32>(p0, 1.0);
    let c1 = rvp * vec4<f32>(p1, 1.0);
    let c2 = rvp * vec4<f32>(p2, 1.0);
    let inv_w = vec3<f32>(
        1.0 / max(c0.w, 1.0e-5),
        1.0 / max(c1.w, 1.0e-5),
        1.0 / max(c2.w, 1.0e-5),
    );
    let s0 = c0.xy * inv_w.x;
    let s1 = c1.xy * inv_w.y;
    let s2 = c2.xy * inv_w.z;

    let det = (s1.x - s0.x) * (s2.y - s0.y) - (s2.x - s0.x) * (s1.y - s0.y);
    var bary = vec3<f32>(0.3333333, 0.3333333, 0.3333334);
    var db_dx = vec3<f32>(0.0);
    var db_dy = vec3<f32>(0.0);
    if abs(det) > 1.0e-12 {
        let inv_det = 1.0 / det;
        let d_l0_dx = (s1.y - s2.y) * inv_det;
        let d_l0_dy = (s2.x - s1.x) * inv_det;
        let d_l1_dx = (s2.y - s0.y) * inv_det;
        let d_l1_dy = (s0.x - s2.x) * inv_det;
        let d_l2_dx = -d_l0_dx - d_l1_dx;
        let d_l2_dy = -d_l0_dy - d_l1_dy;

        let l0 = ((s1.y - s2.y) * (ndc_p.x - s2.x) + (s2.x - s1.x) * (ndc_p.y - s2.y)) * inv_det;
        let l1 = ((s2.y - s0.y) * (ndc_p.x - s2.x) + (s0.x - s2.x) * (ndc_p.y - s2.y)) * inv_det;
        let l2 = 1.0 - l0 - l1;
        let pw = vec3<f32>(l0, l1, l2) * inv_w;
        let sum_pw = pw.x + pw.y + pw.z;
        if abs(sum_pw) > 1.0e-12 {
            bary = pw / sum_pw;
            let px_x = 2.0 / max(vp.x, 1.0);
            let px_y = -2.0 / max(vp.y, 1.0);
            let dpw_dx = vec3<f32>(d_l0_dx, d_l1_dx, d_l2_dx) * inv_w * px_x;
            let dpw_dy = vec3<f32>(d_l0_dy, d_l1_dy, d_l2_dy) * inv_w * px_y;
            db_dx = (dpw_dx - bary * (dpw_dx.x + dpw_dx.y + dpw_dx.z)) / sum_pw;
            db_dy = (dpw_dy - bary * (dpw_dy.x + dpw_dy.y + dpw_dy.z)) / sum_pw;
        }
    }

    let world_pos = p0 * bary.x + p1 * bary.y + p2 * bary.z;
    let n0 = decode_octahedral(nanite_frame_page_heap[b0 + 2u]);
    let n1 = decode_octahedral(nanite_frame_page_heap[b1 + 2u]);
    let n2 = decode_octahedral(nanite_frame_page_heap[b2 + 2u]);
    let local_n = n0 * bary.x + n1 * bary.y + n2 * bary.z;
    let world_n_raw = quat_rotate(inst.rot_quat, local_n);
    let wn_l2 = dot(world_n_raw, world_n_raw);
    let world_n = select(vec3<f32>(0.0, 1.0, 0.0), world_n_raw * inverseSqrt(wn_l2), wn_l2 > 1.0e-8);

    if view_mode != 0u {
        let ndl = 0.42 + 0.58 * clamp(dot(world_n, frame_params.sun_dir_illum.xyz) * 0.5 + 0.5, 0.0, 1.0);
        if view_mode == 1u {
            let col = hash_color(vc.cluster_id ^ (vc.instance_id * 1664525u + 1013904223u));
            return vec4<f32>(col * ndl, 1.0);
        }
        if view_mode == 2u {
            let col = hash_color((vc.cluster_id * 64u + tri_id) ^ (vc.instance_id * 2246822519u));
            return vec4<f32>(col * ndl, 1.0);
        }
        if view_mode == 3u {
            let col = lod_heat_color(cluster_lod(c));
            return vec4<f32>(col * ndl, 1.0);
        }
        if view_mode == 4u {
            let hw_col = vec3<f32>(0.06, 0.80, 0.96);
            let sw_col = vec3<f32>(0.98, 0.52, 0.08);
            return vec4<f32>(select(hw_col, sw_col, use_sw) * ndl, 1.0);
        }
        let page_id = c.cone_and_meta.z;
        let is_root = (c.streaming_meta.z & 1u) != 0u;
        let pcol = select(hash_color(page_id + 17u), vec3<f32>(0.98, 0.25, 0.72), is_root);
        return vec4<f32>(pcol * ndl, 1.0);
    }

    let ca0 = decode_color_ao(w1_0);
    let ca1 = decode_color_ao(w1_1);
    let ca2 = decode_color_ao(w1_2);
    let color_ao = ca0 * bary.x + ca1 * bary.y + ca2 * bary.z;

    let uv0 = decode_uv(nanite_frame_page_heap[b0 + 3u]);
    let uv1 = decode_uv(nanite_frame_page_heap[b1 + 3u]);
    let uv2 = decode_uv(nanite_frame_page_heap[b2 + 3u]);
    let uv_interp = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
    let duv_dx = uv0 * db_dx.x + uv1 * db_dx.y + uv2 * db_dx.z;
    let duv_dy = uv0 * db_dy.x + uv1 * db_dy.y + uv2 * db_dy.z;
    let uv_filter = length(duv_dx) + length(duv_dy);

    let dn_dx = quat_rotate(inst.rot_quat, n0 * db_dx.x + n1 * db_dx.y + n2 * db_dx.z);
    let dn_dy = quat_rotate(inst.rot_quat, n0 * db_dy.x + n1 * db_dy.y + n2 * db_dy.z);
    let normal_var = min(dot(dn_dx, dn_dx) + dot(dn_dy, dn_dy), 0.35);

    let mat_id = cluster_mat(c);
    let lit = shade_pbr(
        world_pos,
        world_n,
        color_ao.rgb,
        clamp(color_ao.a, 0.15, 1.0),
        mat_id,
        uv_interp,
        uv_filter,
        normal_var,
        pos.xy,
    );
    return vec4<f32>(lit, 1.0);
}
