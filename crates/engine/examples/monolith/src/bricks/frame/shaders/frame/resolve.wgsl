import package::generated::frame::FrameUniforms;
import package::generated::frame::GpuMaterialParams;
import package::nanite::frame::nanite_frame_uniforms;
import package::nanite::resolve::nanite_load_depth;
import package::nanite::resolve::nanite_load_vis_hit;
import package::nanite::resolve::nanite_decode_surface;
import package::nanite::resolve::nanite_debug_color;
import package::nanite::resolve::nanite_sample_shadow;

@group(2) @binding(0) var vis_hw_tex: texture_2d<u32>;
@group(2) @binding(1) var<uniform> frame_params: FrameUniforms;
@group(2) @binding(2) var<storage, read> mat_params: array<GpuMaterialParams>;
@group(2) @binding(3) var mat_albedo: texture_2d_array<f32>;
@group(2) @binding(4) var mat_normal: texture_2d_array<f32>;
@group(2) @binding(5) var mat_orm: texture_2d_array<f32>;
@group(2) @binding(6) var pbr_sampler: sampler;
@group(2) @binding(7) var shadow_sampler: sampler_comparison;
@group(2) @binding(8) var gi_sh_r: texture_3d<f32>;
@group(2) @binding(9) var gi_sh_g: texture_3d<f32>;
@group(2) @binding(10) var gi_sh_b: texture_3d<f32>;
@group(2) @binding(11) var gi_sampler: sampler;

const PI: f32 = 3.14159265;
const NO_TEX_LAYER: u32 = 0xFFFFFFFFu;

struct GiPair {
    diff: vec3<f32>,
    refl: vec3<f32>,
}

struct MaterialSample {
    albedo: vec3<f32>,
    shaded_n: vec3<f32>,
    rough: f32,
    metallic: f32,
}

fn temporal_ign(frag_xy: vec2<f32>) -> f32 {
    let f = frame_params.taa_params.w;
    let p = frag_xy + vec2<f32>(5.588238 * f, 13.73219 * f);
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
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
    let mie = (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));
    col += vec3<f32>(1.0, 0.90, 0.72) * (min(mie, 1.8) * 0.22 * (1.0 - up_t * 0.45));

    if include_sun {
        let ct = clamp((cos_theta - 0.992) * 125.0, 0.0, 1.0);
        let corona = ct * ct * ct;
        let st = clamp((cos_theta - 0.94) * (1.0 / 0.06), 0.0, 1.0);
        let st2 = st * st;
        let soft_glow = st2 * st2;
        let disk = smoothstep(0.99992, 0.99997, cos_theta);
        col += frame_params.sun_color_shaft.rgb * (soft_glow * 0.45 + corona * 2.2 + disk * 24.0);
    }
    return col;
}

fn eval_shadow(world_pos: vec3<f32>, geom_n: vec3<f32>, ign: f32) -> f32 {
    if frame_params.gi_max_pad.w < 0.5 {
        return 1.0;
    }
    return nanite_sample_shadow(
        world_pos,
        geom_n,
        shadow_sampler,
        frame_params.sun_dir_illum.xyz,
        ign,
    );
}

fn eval_geomerics_sh_pair(
    sh: vec4<f32>,
    n: vec3<f32>,
    refl: vec3<f32>,
    need_refl: bool,
) -> vec2<f32> {
    let r0 = max(sh.x, 1.0e-5);
    let l1 = sh.yzw;
    let l1_len = length(l1);
    let r = clamp(l1_len / r0, 0.0, 1.0);
    let l1_dir = l1 * (1.0 / max(l1_len, 1.0e-5));
    let p = 1.0 + 2.0 * r;
    let a = (1.0 - r) / (1.0 + r);
    let c1 = (1.0 - a) * (p + 1.0);
    let qn = clamp(0.5 * (1.0 + dot(n, l1_dir)), 0.0, 1.0);
    let dn = max(r0 * (a + c1 * pow(qn, p)), 0.0);
    var dr = dn;
    if need_refl {
        let qr = clamp(0.5 * (1.0 + dot(refl, l1_dir)), 0.0, 1.0);
        dr = max(r0 * (a + c1 * pow(qr, p)), 0.0);
    }
    return vec2<f32>(dn, dr);
}

fn eval_gi_pair(
    world_pos: vec3<f32>,
    geom_n: vec3<f32>,
    n: vec3<f32>,
    refl: vec3<f32>,
    need_refl: bool,
) -> GiPair {
    if frame_params.gi_min_pad.w < 0.5 {
        let up_n = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
        let d = mix(vec3<f32>(0.24, 0.19, 0.14), vec3<f32>(0.36, 0.44, 0.58), up_n);
        var r_col = d;
        if need_refl {
            let up_r = clamp(refl.y * 0.5 + 0.5, 0.0, 1.0);
            r_col = mix(vec3<f32>(0.24, 0.19, 0.14), vec3<f32>(0.36, 0.44, 0.58), up_r);
        }
        return GiPair(d * 1.55, r_col);
    }
    let gmin = frame_params.gi_min_pad.xyz;
    let gmax = frame_params.gi_max_pad.xyz;
    let biased = world_pos + geom_n * 0.30;
    let uvw_raw = (biased - gmin) / max(gmax - gmin, vec3<f32>(1.0e-3));
    let dims = vec3<f32>(textureDimensions(gi_sh_r));
    let half_tex = vec3<f32>(0.5) / max(dims, vec3<f32>(1.0));
    let uvw = clamp(uvw_raw, half_tex, vec3<f32>(1.0) - half_tex);

    let gr = eval_geomerics_sh_pair(
        textureSampleLevel(gi_sh_r, gi_sampler, uvw, 0.0),
        n,
        refl,
        need_refl,
    );
    let gg = eval_geomerics_sh_pair(
        textureSampleLevel(gi_sh_g, gi_sampler, uvw, 0.0),
        n,
        refl,
        need_refl,
    );
    let gb = eval_geomerics_sh_pair(
        textureSampleLevel(gi_sh_b, gi_sampler, uvw, 0.0),
        n,
        refl,
        need_refl,
    );
    return GiPair(vec3<f32>(gr.x, gg.x, gb.x) * 1.55, vec3<f32>(gr.y, gg.y, gb.y));
}

fn eval_contact_ao(px_i: vec2<i32>, size: vec2<u32>, center_d: f32, ign: f32) -> f32 {
    if center_d <= 1.0e-5 {
        return 1.0;
    }
    let near_z = nanite_frame_uniforms.cull_pos_near.w;
    let center_z = near_z / center_d;
    let max_rad = clamp(20.0 / max(center_z, 1.2), 3.5, 14.0);
    let max_px = vec2<i32>(size) - vec2<i32>(1);
    let a0 = ign * 6.2831853;
    var dir = vec2<f32>(cos(a0), sin(a0));
    let ign_phase = ign * 7.31;
    var occ = 0.0;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let ring = select(0.48, 0.96, (k & 1u) != 0u);
        let rad = max_rad * (ring + 0.14 * fract(ign_phase + f32(k) * 0.37));
        let off = vec2<i32>(round(dir * rad));
        let c = clamp(px_i + off, vec2<i32>(0), max_px);
        let sd = nanite_load_depth(c, size.x);
        if sd > 1.0e-5 {
            let dz = center_z - near_z / sd;
            occ += smoothstep(0.02, 0.16, dz) * (1.0 - smoothstep(0.38, 1.15, dz));
        }
        dir = vec2<f32>(-dir.y, dir.x);
    }
    return clamp(1.0 - occ * 0.17, 0.42, 1.0);
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
    let x = clamp(1.0 - cos_t, 0.0, 1.0);
    let x2 = x * x;
    let f = x2 * x2 * x;
    return f0 + (vec3<f32>(1.0) - f0) * f;
}

fn eval_lantern_glow(world_pos: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>, metallic: f32) -> vec3<f32> {
    if world_pos.x < -21.0 || world_pos.x > 13.5 || world_pos.y > 11.5 || abs(world_pos.z) > 12.0 {
        return vec3<f32>(0.0);
    }
    var accum = 0.0;
    for (var i = 0u; i < 5u; i = i + 1u) {
        var lp = vec3<f32>(-13.8, 3.6, 0.0);
        if i > 0u {
            let lx = select(-1.8, 6.2, i >= 3u);
            let lz = select(-4.8, 4.8, (i & 1u) == 0u);
            lp = vec3<f32>(lx, 4.1, lz);
        }
        let delta = lp - world_pos;
        let d2 = dot(delta, delta);
        if d2 < 46.0 {
            let inv_d = inverseSqrt(max(d2, 0.08));
            let atten = 1.0 - d2 * (1.0 / 46.0);
            let falloff = (atten * atten) / (1.0 + d2 * 0.45);
            let ndl = clamp(dot(n, delta) * inv_d * 0.72 + 0.28, 0.0, 1.0);
            accum += falloff * ndl;
        }
    }
    return vec3<f32>(2.1, 1.344, 0.588) * albedo * (accum * (1.0 - metallic * 0.7) * (1.0 / PI));
}

fn apply_aerial_fog(
    color: vec3<f32>,
    world_pos: vec3<f32>,
    cam_pos: vec3<f32>,
) -> vec3<f32> {
    let to_w = world_pos - cam_pos;
    let dist = length(to_w);
    let view_dir = to_w / max(dist, 1.0e-4);
    let density = frame_params.fog_params.x;
    let falloff = frame_params.fog_params.y;
    let avg_h = max(0.5 * (cam_pos.y + world_pos.y), 0.0);
    let optical = density * dist * exp(-falloff * avg_h);
    let fog_amt = 1.0 - exp(-min(optical, 0.30));
    let sun_dot = clamp(dot(view_dir, frame_params.sun_dir_illum.xyz), 0.0, 1.0);
    let fog_col = mix(frame_params.fog_color.rgb, frame_params.sky_horiz_sun.rgb, sun_dot * 0.45);
    return mix(color, fog_col, clamp(fog_amt, 0.0, 0.22));
}

fn apply_normal_map(n: vec3<f32>, t_frame: vec4<f32>, nm_sample: vec2<f32>) -> vec3<f32> {
    if t_frame.w == 0.0 {
        return n;
    }
    let b = cross(n, t_frame.xyz) * t_frame.w;
    let tn_xy = nm_sample * 2.0 - vec2<f32>(1.0);
    let tn_z = sqrt(max(1.0 - dot(tn_xy, tn_xy), 0.01));
    let world_nm = t_frame.xyz * tn_xy.x + b * tn_xy.y + n * tn_z;
    let l2 = dot(world_nm, world_nm);
    return select(n, world_nm * inverseSqrt(l2), l2 > 1.0e-6);
}

fn eval_material(
    mat_id: u32,
    uv_interp: vec2<f32>,
    duv_dx: vec2<f32>,
    duv_dy: vec2<f32>,
    t_frame: vec4<f32>,
    geom_n: vec3<f32>,
    ln_l2: f32,
) -> MaterialSample {
    let layer_flags = mat_params[mat_id].layer_flags;
    let base_cr = mat_params[mat_id].base_color_rough;
    let tex_layer = layer_flags.x;
    let min_mip = layer_flags.z;

    var albedo = base_cr.rgb;
    var rough = base_cr.w;
    var metallic = mat_params[mat_id].emissive_metal.w;
    var shaded_n = geom_n;

    if tex_layer != NO_TEX_LAYER {
        let li = i32(tex_layer);
        let grad_x = duv_dx * 0.70710678;
        let grad_y = duv_dy * 0.70710678;
        var nm_xy = vec2<f32>(0.5, 0.5);
        var orm_gb = vec2<f32>(0.8, 0.0);
        if min_mip == 0u {
            albedo = textureSampleGrad(mat_albedo, pbr_sampler, uv_interp, li, grad_x, grad_y).rgb;
            nm_xy = textureSampleGrad(mat_normal, pbr_sampler, uv_interp, li, grad_x, grad_y).rg;
            orm_gb = textureSampleGrad(mat_orm, pbr_sampler, uv_interp, li, grad_x, grad_y).gb;
        } else {
            let max_g = sqrt(max(dot(grad_x, grad_x), dot(grad_y, grad_y)));
            let dim_f = f32(textureDimensions(mat_albedo).x);
            let lod = max(log2(max(max_g * dim_f, 1.0)), f32(min_mip));
            albedo = textureSampleLevel(mat_albedo, pbr_sampler, uv_interp, li, lod).rgb;
            nm_xy = textureSampleLevel(mat_normal, pbr_sampler, uv_interp, li, lod).rg;
            if min_mip == 1u {
                orm_gb = textureSampleGrad(mat_orm, pbr_sampler, uv_interp, li, grad_x, grad_y).gb;
            } else {
                let orm_dim = f32(textureDimensions(mat_orm).x);
                let orm_lod = max(log2(max(max_g * orm_dim, 1.0)), f32(min_mip - 1u));
                orm_gb = textureSampleLevel(mat_orm, pbr_sampler, uv_interp, li, orm_lod).gb;
            }
        }
        rough = clamp(orm_gb.x * base_cr.w, 0.04, 0.98);
        metallic = clamp(orm_gb.y * metallic, 0.0, 1.0);
        shaded_n = apply_normal_map(geom_n, t_frame, nm_xy);
    }

    let normal_var = clamp(1.0 - ln_l2, 0.0, 0.30);
    rough = sqrt(clamp(rough * rough + normal_var * 0.35, 0.03, 0.96));
    return MaterialSample(albedo, shaded_n, rough, metallic);
}

fn shade_sponza_pbr(
    world_pos: vec3<f32>,
    geom_n: vec3<f32>,
    n: vec3<f32>,
    albedo: vec3<f32>,
    rough: f32,
    metallic: f32,
    ao: f32,
    mat_id: u32,
    ign: f32,
) -> vec3<f32> {
    let l = frame_params.sun_dir_illum.xyz;
    let flags = mat_params[mat_id].layer_flags.y;
    let is_fabric = (flags & 2u) != 0u;
    let is_foliage = (flags & 4u) != 0u;
    let is_glass = (flags & 8u) != 0u;

    let raw_ndl = dot(n, l);
    let ndl = max(raw_ndl, 0.0);
    var sh = 0.0;
    if raw_ndl > -0.25 || is_fabric || is_foliage {
        sh = eval_shadow(world_pos, geom_n, ign);
    }

    let cam_pos = nanite_frame_uniforms.render_pos_time.xyz;
    let to_cam = cam_pos - world_pos;
    let v = to_cam / max(length(to_cam), 1.0e-4);
    let sun_rad = frame_params.sun_color_shaft.rgb * frame_params.sun_dir_illum.w;
    let ndv = max(dot(n, v), 0.001);
    let h_raw = l + v;
    let h_l2 = dot(h_raw, h_raw);
    let h = select(n, h_raw * inverseSqrt(h_l2), h_l2 > 1.0e-6);
    let ndh = max(dot(n, h), 0.0);
    let vdh = max(dot(v, h), 0.0);

    let f0_d = mix(vec3<f32>(0.04), albedo, metallic);
    let d_term = distribution_ggx(ndh, rough);
    let g_term = geometry_smith(ndv, ndl, rough);
    let f_term = fresnel_schlick(vdh, f0_d);
    let spec = (d_term * g_term * f_term) / max(4.0 * ndv * ndl, 1.0e-4);
    let kd = (vec3<f32>(1.0) - f_term) * (1.0 - metallic);
    let diff = kd * albedo * (1.0 / PI);

    var lit = (diff + spec) * sun_rad * (ndl * sh) + mat_params[mat_id].emissive_metal.rgb;

    if is_fabric {
        let back_n = max(dot(-n, l), 0.0);
        let bv = clamp(dot(-v, l), 0.0, 1.0);
        let back_view = bv * bv * sqrt(bv);
        let st = clamp(1.0 - ndv, 0.0, 1.0);
        let sheen = st * st * st * 0.22;
        let trans = (back_n * 0.42 + back_view * 0.55) * (0.30 + 0.70 * sh);
        lit += albedo * sun_rad * (trans * 0.36 * (1.0 / PI)) + albedo * (sheen * (0.25 + 0.75 * sh));
    }
    if is_foliage {
        let back_n = max(dot(-n, l), 0.0);
        let bv = clamp(dot(-v, l), 0.0, 1.0);
        let back_view = bv * bv * bv;
        let sss_col = albedo * vec3<f32>(1.15, 1.35, 0.65);
        let trans = (back_n * 0.35 + back_view * 0.60) * (0.30 + 0.70 * sh);
        lit += sss_col * sun_rad * (trans * 0.42 * (1.0 / PI));
    }
    lit += eval_lantern_glow(world_pos, n, albedo, metallic);

    let refl = reflect(-v, n);
    let need_refl = rough <= 0.72 || metallic >= 0.05 || is_glass;
    let diff_w = albedo * ((1.0 - metallic * 0.82) * ao);
    let ao_spec = clamp(pow(ndv + ao, exp2(-16.0 * rough - 1.0)) - 1.0 + ao, 0.0, 1.0);
    let f_env = fresnel_schlick(ndv, f0_d);
    let spec_boost = select(0.65 + 0.85 * metallic, 1.40, is_glass);
    let env_w = f_env * ((1.0 - rough * 0.70) * ao_spec * spec_boost);
    let open_z = 1.0 - smoothstep(2.8, 7.6, abs(world_pos.z));
    let open_x = 1.0 - smoothstep(9.5, 16.5, abs(world_pos.x - 1.0));
    let sky_w = clamp(open_z * open_x * smoothstep(-0.1, 0.55, refl.y), 0.0, 1.0) * (1.0 - rough * 0.7);

    let gi = eval_gi_pair(world_pos, geom_n, n, refl, need_refl);
    lit += gi.diff * diff_w;

    var env_col = gi.refl;
    if sky_w > 0.005 {
        env_col = mix(gi.refl, eval_sky(refl, false) * 0.72, sky_w);
    }
    lit += env_col * env_w;
    return apply_aerial_fog(lit, world_pos, cam_pos);
}

@fragment
fn fs_resolve(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let vp = max(nanite_frame_uniforms.viewport_lod.xy, vec2<f32>(1.0));
    let size = max(vec2<u32>(nanite_frame_uniforms.viewport_lod.xy), vec2<u32>(1u));
    let px = min(vec2<u32>(pos.xy), size - vec2<u32>(1u));
    let view_mode = nanite_frame_uniforms.flags.z;
    let hit = nanite_load_vis_hit(vis_hw_tex, px, size.x);

    let cam_pos = nanite_frame_uniforms.render_pos_time.xyz;
    let inv_vp = nanite_frame_uniforms.inv_render_vp;
    if hit.vis_id == 0u {
        if view_mode != 0u {
            return vec4<f32>(0.05, 0.055, 0.07, 0.0);
        }
        let uv_s = pos.xy / vp;
        let ndc_s = vec2<f32>(uv_s.x * 2.0 - 1.0, 1.0 - uv_s.y * 2.0);
        let far_h = inv_vp * vec4<f32>(ndc_s, 0.0001, 1.0);
        let far_r = (far_h.xyz - cam_pos * far_h.w) * select(-1.0, 1.0, far_h.w >= 0.0);
        return vec4<f32>(eval_sky(normalize(far_r), true), 0.0);
    }

    let surf = nanite_decode_surface(hit, pos.xy, vp);
    if view_mode != 0u {
        return vec4<f32>(
            nanite_debug_color(view_mode, surf, frame_params.sun_dir_illum.xyz),
            1.0,
        );
    }

    let ign = temporal_ign(pos.xy);
    let contact_ao = eval_contact_ao(vec2<i32>(px), size, hit.depth, ign);
    let mat_id = min(surf.material_id, 31u);
    let ao = clamp(0.45 + 0.55 * surf.vertex_ao, 0.35, 1.0) * contact_ao;
    let mat = eval_material(
        mat_id,
        surf.uv,
        surf.duv_dx,
        surf.duv_dy,
        surf.tangent,
        surf.geom_n,
        surf.normal_len2,
    );

    let lit = shade_sponza_pbr(
        surf.world_pos,
        surf.geom_n,
        mat.shaded_n,
        mat.albedo,
        mat.rough,
        mat.metallic,
        ao,
        mat_id,
        ign,
    );
    return vec4<f32>(lit, 1.0);
}
