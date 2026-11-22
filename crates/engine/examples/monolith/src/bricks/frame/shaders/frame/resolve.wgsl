import package::generated::frame::FrameUniforms;
import package::generated::frame::GpuMaterialParams;
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
import package::nanite::common::decode_pos;
import package::nanite::common::decode_color_ao;
import package::nanite::common::decode_octahedral;
import package::nanite::common::decode_uv;

@group(2) @binding(0) var vis_hw_tex: texture_2d<u32>;
@group(2) @binding(1) var hw_depth_tex: texture_depth_2d;
@group(2) @binding(2) var<uniform> frame_params: FrameUniforms;
@group(2) @binding(3) var<storage, read> mat_params: array<GpuMaterialParams>;
@group(2) @binding(4) var mat_albedo: texture_2d_array<f32>;
@group(2) @binding(5) var mat_normal: texture_2d_array<f32>;
@group(2) @binding(6) var mat_orm: texture_2d_array<f32>;
@group(2) @binding(7) var pbr_sampler: sampler;

const PI: f32 = 3.14159265;
const NO_TEX_LAYER: u32 = 0xFFFFFFFFu;

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
    let t = clamp(f32(lod) * 0.08, 0.0, 1.0);
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

fn eval_sky(dir: vec3<f32>, include_sun: bool) -> vec3<f32> {
    let sun_dir = frame_params.sun_dir_illum.xyz;
    let up_t = pow(clamp(dir.y, 0.0, 1.0), 0.45);
    let sun_az = clamp(
        dot(
            normalize(vec2<f32>(dir.x, dir.z) + vec2<f32>(1.0e-5, 0.0)),
            normalize(vec2<f32>(sun_dir.x, sun_dir.z) + vec2<f32>(1.0e-5, 0.0)),
        ) * 0.5
            + 0.5,
        0.0,
        1.0,
    );
    let horiz = mix(frame_params.sky_horiz_opp.rgb, frame_params.sky_horiz_sun.rgb, pow(sun_az, 1.4));
    var col = mix(horiz, frame_params.sky_zenith.rgb, up_t) * 1.45;

    if dir.y < 0.0 {
        let ground_fade = clamp(-dir.y * 4.5, 0.0, 1.0);
        col = mix(horiz * 1.2, vec3<f32>(0.38, 0.33, 0.27), ground_fade);
    }

    let cos_theta = dot(dir, sun_dir);
    let g = frame_params.fog_color.w;
    let g2 = g * g;
    let denom = max(1.0 + g2 - 2.0 * g * cos_theta, 0.02);
    let mie = (1.0 - g2) / (4.0 * PI * pow(denom, 1.5));
    col += vec3<f32>(1.0, 0.90, 0.72) * (min(mie, 1.8) * 0.22 * (1.0 - up_t * 0.45));

    if include_sun {
        let corona = pow(clamp((cos_theta - 0.992) / 0.008, 0.0, 1.0), 3.0);
        let soft_glow = pow(clamp((cos_theta - 0.94) / 0.06, 0.0, 1.0), 4.0);
        let disk = smoothstep(0.99992, 0.99997, cos_theta);
        col += frame_params.sun_color_shaft.rgb * (soft_glow * 0.45 + corona * 2.2 + disk * 24.0);
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
    let jx = cos(rot_angle) * radius_px * 0.55;
    let jy = sin(rot_angle) * radius_px * 0.55;
    let cmp_z = clip.z + depth_bias;
    let coverage = smoothstep(0.98, 0.80, margin);
    let lit = sample_shadow_bilinear(sx + jx, sy + jy, cmp_z, dim_u, dim_i, base_off);
    return vec2<f32>(lit, coverage);
}

fn eval_shadow(world_pos: vec3<f32>, geom_n: vec3<f32>, frag_xy: vec2<f32>) -> f32 {
    let ign = fract(52.9829189 * fract(dot(frag_xy, vec2<f32>(0.06711056, 0.00583715))));
    let angle = ign * 6.2831853;
    let ndl = max(dot(geom_n, frame_params.sun_dir_illum.xyz), 0.0);
    let slope = clamp(1.0 - ndl, 0.0, 1.0);
    let c0 = sample_cascade_pcf(
        world_pos,
        geom_n,
        nanite_frame_uniforms.shadow_vp0,
        SHADOW_C0_DIM,
        0u,
        0.022 + slope * 0.038,
        0.00055 + slope * 0.00080,
        angle,
        0.85,
    );
    if c0.y >= 0.99 {
        return c0.x;
    }
    let c1 = sample_cascade_pcf(
        world_pos,
        geom_n,
        nanite_frame_uniforms.shadow_vp1,
        SHADOW_C1_DIM,
        SHADOW_C0_DIM * SHADOW_C0_DIM,
        0.028 + slope * 0.048,
        0.00060 + slope * 0.00085,
        angle,
        0.95,
    );
    let far_val = mix(1.0, c1.x, c1.y);
    return mix(far_val, c0.x, c0.y);
}

fn eval_contact_ao(px_i: vec2<i32>, size: vec2<u32>, center_d: f32, frag_xy: vec2<f32>) -> f32 {
    if center_d <= 1.0e-5 {
        return 1.0;
    }
    let near_z = nanite_frame_uniforms.cull_pos_near.w;
    let center_z = near_z / center_d;
    let radius = clamp(16.0 / max(center_z, 1.5), 3.0, 11.0);
    let max_px = vec2<i32>(size) - vec2<i32>(1);
    let ign = fract(52.9829189 * fract(dot(frag_xy, vec2<f32>(0.06711056, 0.00583715))));
    let a0 = ign * 6.2831853;
    var occ = 0.0;
    for (var k = 0; k < 3; k = k + 1) {
        let ang = a0 + f32(k) * 2.0943951;
        let off = vec2<i32>(round(vec2<f32>(cos(ang), sin(ang)) * radius));
        let c = clamp(px_i + off, vec2<i32>(0), max_px);
        let sd = textureLoad(hw_depth_tex, c, 0);
        if sd > 1.0e-5 {
            let sz = near_z / sd;
            let dz = center_z - sz;
            let hit = smoothstep(0.025, 0.14, dz) * (1.0 - smoothstep(0.32, 0.85, dz));
            occ += hit;
        }
    }
    return clamp(1.0 - occ * 0.21, 0.42, 1.0);
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

fn eval_lantern_glow(world_pos: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>, metallic: f32) -> vec3<f32> {
    let lantern_col = vec3<f32>(1.0, 0.64, 0.28) * 2.25;
    var accum = vec3<f32>(0.0);
    let xs = array<f32, 5>(-13.8, -1.8, -1.8, 6.2, 6.2);
    let ys = array<f32, 5>(3.6, 4.1, 4.1, 4.1, 4.1);
    let zs = array<f32, 5>(0.0, -4.8, 4.8, -4.8, 4.8);
    for (var i = 0; i < 5; i = i + 1) {
        let lp = vec3<f32>(xs[i], ys[i], zs[i]);
        let delta = lp - world_pos;
        let d2 = dot(delta, delta);
        if d2 < 46.0 {
            let inv_d = inverseSqrt(max(d2, 0.08));
            let ldir = delta * inv_d;
            let atten = clamp(1.0 - d2 * (1.0 / 46.0), 0.0, 1.0);
            let falloff = (atten * atten) / (1.0 + d2 * 0.45);
            let ndl = clamp(dot(n, ldir) * 0.72 + 0.28, 0.0, 1.0);
            accum += lantern_col * (falloff * ndl);
        }
    }
    return accum * albedo * (1.0 - metallic * 0.7) * (1.0 / PI);
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
    let avg_h = max(0.5 * (cam_pos.y + world_pos.y), 0.0);
    let optical = density * dist * exp(-falloff * avg_h);
    let fog_amt = 1.0 - exp(-min(optical, 0.35));
    let sun_dot = clamp(dot(view_dir, frame_params.sun_dir_illum.xyz), 0.0, 1.0);
    let fog_col = mix(frame_params.fog_color.rgb, frame_params.sky_horiz_sun.rgb, sun_dot * 0.45);
    return mix(color, fog_col, clamp(fog_amt, 0.0, 0.25));
}

fn perturb_normal(
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    uv2: vec2<f32>,
    n: vec3<f32>,
    nm_sample: vec3<f32>,
) -> vec3<f32> {
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let du1 = uv1 - uv0;
    let du2 = uv2 - uv0;
    let det = du1.x * du2.y - du1.y * du2.x;
    if abs(det) <= 1.0e-8 {
        return n;
    }
    let inv_det = 1.0 / det;
    let t_raw = (e1 * du2.y - e2 * du1.y) * inv_det;
    let b_raw = (e2 * du1.x - e1 * du2.x) * inv_det;
    let t_ortho = t_raw - n * dot(n, t_raw);
    let t_l2 = dot(t_ortho, t_ortho);
    if t_l2 <= 1.0e-8 {
        return n;
    }
    let t = t_ortho * inverseSqrt(t_l2);
    let sign_b = select(-1.0, 1.0, dot(cross(n, t), b_raw) >= 0.0);
    let b = cross(n, t) * sign_b;
    let tn_xy = nm_sample.xy * 2.0 - vec2<f32>(1.0);
    let tn_z = sqrt(max(1.0 - dot(tn_xy, tn_xy), 0.01));
    let world_nm = t * tn_xy.x + b * tn_xy.y + n * tn_z;
    let l2 = dot(world_nm, world_nm);
    return select(n, world_nm * inverseSqrt(l2), l2 > 1.0e-6);
}

fn shade_sponza_pbr(
    world_pos: vec3<f32>,
    geom_n: vec3<f32>,
    shaded_n: vec3<f32>,
    albedo: vec3<f32>,
    rough_in: f32,
    metallic: f32,
    emissive: vec3<f32>,
    ao: f32,
    flags: u32,
    normal_var: f32,
    frag_xy: vec2<f32>,
) -> vec3<f32> {
    let cam_pos = nanite_frame_uniforms.render_pos_time.xyz;
    let to_cam = cam_pos - world_pos;
    let v_len = max(length(to_cam), 1.0e-4);
    let v = to_cam / v_len;
    let n = shaded_n;
    let l = frame_params.sun_dir_illum.xyz;
    let sun_rad = frame_params.sun_color_shaft.rgb * frame_params.sun_dir_illum.w;
    let rough = sqrt(clamp(rough_in * rough_in + normal_var * 0.35, 0.03, 0.96));

    let is_fabric = (flags & 2u) != 0u;
    let is_foliage = (flags & 4u) != 0u;
    let is_glass = (flags & 8u) != 0u;

    let raw_ndl = dot(n, l);
    let ndl = max(raw_ndl, 0.0);
    var sh = 0.0;
    if raw_ndl > -0.25 || is_fabric || is_foliage {
        sh = eval_shadow(world_pos, geom_n, frag_xy);
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

    if is_fabric {
        let back_n = max(dot(-n, l), 0.0);
        let back_view = pow(clamp(dot(-v, l), 0.0, 1.0), 2.5);
        let sheen = pow(clamp(1.0 - ndv, 0.0, 1.0), 3.0) * 0.22;
        let trans = (back_n * 0.42 + back_view * 0.55) * (0.30 + 0.70 * sh);
        direct += albedo * sun_rad * (trans * 0.36 * (1.0 / PI)) + albedo * sheen * (0.25 + 0.75 * sh);
    }
    if is_foliage {
        let back_n = max(dot(-n, l), 0.0);
        let back_view = pow(clamp(dot(-v, l), 0.0, 1.0), 3.0);
        let sss_col = albedo * vec3<f32>(1.15, 1.35, 0.65);
        let trans = (back_n * 0.35 + back_view * 0.60) * (0.30 + 0.70 * sh);
        direct += sss_col * sun_rad * (trans * 0.42 * (1.0 / PI));
    }

    let open_z = 1.0 - smoothstep(2.8, 7.6, abs(world_pos.z));
    let open_x = 1.0 - smoothstep(9.5, 16.5, abs(world_pos.x - 1.0));
    let sky_vis = clamp(0.28 + 0.72 * open_z * open_x + smoothstep(6.0, 14.0, world_pos.y) * 0.25, 0.24, 1.0);

    let sky_col = mix(vec3<f32>(0.58, 0.68, 0.84), vec3<f32>(0.34, 0.54, 0.88), clamp(n.y * 0.5 + 0.5, 0.0, 1.0)) * (1.02 * sky_vis);
    let floor_bounce = vec3<f32>(0.88, 0.72, 0.52) * (0.76 * (0.55 + 0.45 * open_z));
    let wall_bounce = vec3<f32>(0.76, 0.63, 0.48) * (0.44 * (1.0 - abs(n.y)));
    let sun_opp_bounce = vec3<f32>(0.94, 0.78, 0.56) * (clamp(dot(n, -l), 0.0, 1.0) * 0.28 * open_z);
    let hemi = mix(floor_bounce, sky_col, n.y * 0.5 + 0.5) + wall_bounce * 0.48 + sun_opp_bounce;

    let ao_spec = clamp(pow(ndv + ao, exp2(-16.0 * rough - 1.0)) - 1.0 + ao, 0.0, 1.0);
    let refl = reflect(-v, n);
    let sky_refl = eval_sky(refl, false);
    let interior_refl = mix(vec3<f32>(0.32, 0.27, 0.22), vec3<f32>(0.58, 0.52, 0.44), clamp(refl.y * 0.5 + 0.5, 0.0, 1.0));
    let env_col = mix(interior_refl, sky_refl * 0.65, clamp(sky_vis * smoothstep(-0.15, 0.55, refl.y), 0.0, 1.0));
    let f_env = fresnel_schlick(ndv, f0);
    let spec_boost = select(0.55 + 0.85 * metallic, 1.35, is_glass);
    let env_spec = env_col * f_env * (1.0 - rough * 0.72) * ao_spec * spec_boost;

    let indirect = hemi * albedo * (1.0 - metallic * 0.78) * ao + env_spec;
    let lanterns = eval_lantern_glow(world_pos, n, albedo, metallic);

    let lit = direct + indirect + lanterns + emissive;
    return apply_aerial_fog(lit, world_pos, cam_pos, -v);
}

fn ray_tri_bary(o: vec3<f32>, e1: vec3<f32>, e2: vec3<f32>, q: vec3<f32>, r: vec3<f32>) -> vec3<f32> {
    let h = cross(r, e2);
    let det = dot(e1, h);
    if abs(det) <= 1.0e-12 {
        return vec3<f32>(0.3333333, 0.3333333, 0.3333334);
    }
    let inv_det = 1.0 / det;
    let u = dot(o, h) * inv_det;
    let v = dot(r, q) * inv_det;
    return vec3<f32>(1.0 - u - v, u, v);
}

fn world_ray(inv_vp: mat4x4<f32>, ndc: vec2<f32>, cam_pos: vec3<f32>) -> vec3<f32> {
    let h = inv_vp * vec4<f32>(ndc, 0.5, 1.0);
    let w = h.xyz / max(abs(h.w), 1.0e-5) * sign(h.w);
    return w - cam_pos;
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

    let vis_slot = (vis_id >> 6u) - 1u;
    let tri_id = vis_id & 63u;
    let vc = nanite_frame_vis_clusters[vis_slot];
    let c_meta = nanite_frame_clusters[vc.cluster_id].cone_and_meta;
    let c_min = nanite_frame_clusters[vc.cluster_id].aabb_min.xyz;
    let c_ext = nanite_frame_clusters[vc.cluster_id].aabb_extent.xyz;
    let inst_ps = nanite_frame_instances[vc.instance_id].pos_scale;
    let inst_rq = nanite_frame_instances[vc.instance_id].rot_quat;
    let vert_count = c_meta.y & 0xFFu;
    let page_base = vc.heap_slot * PAGE_WORDS + c_meta.w;
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

    let p0 = inst_ps.xyz + quat_rotate(inst_rq, decode_pos(w0_0, w1_0, c_min, c_ext) * inst_ps.w);
    let p1 = inst_ps.xyz + quat_rotate(inst_rq, decode_pos(w0_1, w1_1, c_min, c_ext) * inst_ps.w);
    let p2 = inst_ps.xyz + quat_rotate(inst_rq, decode_pos(w0_2, w1_2, c_min, c_ext) * inst_ps.w);

    let cam_pos = nanite_frame_uniforms.render_pos_time.xyz;
    let inv_vp = nanite_frame_uniforms.inv_render_vp;
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let o = cam_pos - p0;
    let q = cross(o, e1);
    let r_c = world_ray(inv_vp, ndc_p, cam_pos);
    let r_x = world_ray(inv_vp, ndc_p + vec2<f32>(2.0 / max(vp.x, 1.0), 0.0), cam_pos);
    let r_y = world_ray(inv_vp, ndc_p + vec2<f32>(0.0, -2.0 / max(vp.y, 1.0)), cam_pos);
    let bary = ray_tri_bary(o, e1, e2, q, r_c);
    let bary_x = ray_tri_bary(o, e1, e2, q, r_x);
    let bary_y = ray_tri_bary(o, e1, e2, q, r_y);
    let db_dx = bary_x - bary;
    let db_dy = bary_y - bary;

    let world_pos = p0 * bary.x + p1 * bary.y + p2 * bary.z;
    let n0 = decode_octahedral(nanite_frame_page_heap[b0 + 2u]);
    let n1 = decode_octahedral(nanite_frame_page_heap[b1 + 2u]);
    let n2 = decode_octahedral(nanite_frame_page_heap[b2 + 2u]);
    let local_n = n0 * bary.x + n1 * bary.y + n2 * bary.z;
    let ln_l2 = dot(local_n, local_n);
    let world_n_raw = quat_rotate(inst_rq, local_n);
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
            let col = lod_heat_color((c_meta.y >> 16u) & 0xFFu);
            return vec4<f32>(col * ndl, 1.0);
        }
        if view_mode == 4u {
            let hw_col = vec3<f32>(0.06, 0.80, 0.96);
            let sw_col = vec3<f32>(0.98, 0.52, 0.08);
            return vec4<f32>(select(hw_col, sw_col, use_sw) * ndl, 1.0);
        }
        let page_id = c_meta.z;
        let is_root = (nanite_frame_clusters[vc.cluster_id].streaming_meta.z & 1u) != 0u;
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
    var duv_dx = uv0 * db_dx.x + uv1 * db_dx.y + uv2 * db_dx.z;
    var duv_dy = uv0 * db_dy.x + uv1 * db_dy.y + uv2 * db_dy.z;
    let dx_len = length(duv_dx);
    if dx_len > 0.25 {
        duv_dx *= 0.25 / dx_len;
    }
    let dy_len = length(duv_dy);
    if dy_len > 0.25 {
        duv_dy *= 0.25 / dy_len;
    }

    let v_dir = normalize(cam_pos - world_pos);
    let geom_n = select(world_n, -world_n, dot(world_n, v_dir) < 0.0);

    let mat_id = min((c_meta.y >> 24u) & 0xFFu, 31u);
    let mp = mat_params[mat_id];
    let tex_layer = mp.layer_flags.x;
    let mat_flags = mp.layer_flags.y;

    var albedo = mp.base_color_rough.rgb;
    var rough = mp.base_color_rough.w;
    var metallic = mp.emissive_metal.w;
    var shaded_n = geom_n;

    if tex_layer != NO_TEX_LAYER {
        let li = i32(tex_layer);
        albedo = textureSampleGrad(mat_albedo, pbr_sampler, uv_interp, li, duv_dx, duv_dy).rgb;
        let nm_raw = textureSampleGrad(mat_normal, pbr_sampler, uv_interp, li, duv_dx, duv_dy).rgb;
        let orm = textureSampleGrad(mat_orm, pbr_sampler, uv_interp, li, duv_dx, duv_dy).rgb;
        rough = clamp(orm.g * mp.base_color_rough.w, 0.04, 0.98);
        metallic = clamp(orm.b * mp.emissive_metal.w, 0.0, 1.0);
        shaded_n = perturb_normal(p0, p1, p2, uv0, uv1, uv2, geom_n, nm_raw);
    }

    let vert_ao = clamp(0.45 + 0.55 * color_ao.a, 0.35, 1.0);
    let ss_ao = eval_contact_ao(px_i, size, win_depth, pos.xy);
    let ao = vert_ao * ss_ao;
    let normal_var = clamp(1.0 - ln_l2, 0.0, 0.30);

    let lit = shade_sponza_pbr(
        world_pos,
        geom_n,
        shaded_n,
        albedo,
        rough,
        metallic,
        mp.emissive_metal.rgb,
        ao,
        mat_flags,
        normal_var,
        pos.xy,
    );
    return vec4<f32>(lit, 1.0);
}
