import package::engine::view::view_clip;
import package::generated::world::WorldUniforms;
import package::generated::world::VisibleBlade;
import package::world::common::aerial_fog_factor;

@group(2) @binding(0) var<uniform> world: WorldUniforms;
@group(2) @binding(1) var<storage, read> vis_blades: array<VisibleBlade>;

struct GrassNearOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) lit_t: vec4<f32>,
    @location(1) @interpolate(flat) flower_col_flag: vec4<f32>,
}

struct GrassFarOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) lit_t: vec4<f32>,
    @location(1) @interpolate(flat) mid_corr: vec3<f32>,
}

fn flower_albedo(flower_code: f32) -> vec3<f32> {
    if flower_code > 3.5 {
        return vec3<f32>(0.95, 0.22, 0.16);
    }
    if flower_code > 2.5 {
        return vec3<f32>(0.36, 0.58, 0.98);
    }
    if flower_code > 1.5 {
        return vec3<f32>(0.98, 0.94, 0.86);
    }
    return vec3<f32>(0.98, 0.76, 0.14);
}

fn eval_grass_light(
    t: f32,
    albedo: vec3<f32>,
    n: vec3<f32>,
    sun_dir: vec3<f32>,
    shadow: f32,
    gust_t: f32,
    back_scatter: f32,
    sss_tint: vec3<f32>,
    sky_top: vec3<f32>,
    dyn_light: vec3<f32>,
) -> vec3<f32> {
    let ao = mix(0.14, 1.0, t * (1.42 - 0.42 * t));
    let ndl = clamp((dot(n, sun_dir) + 0.36) * 0.7352941, 0.0, 1.0);
    let wave_band = mix(0.62, 1.48, gust_t * (0.25 + 0.75 * t));
    let sss_mask = t * t * (0.16 + 0.84 * back_scatter) * shadow;
    let sss = world.sun_color.rgb * sss_tint * sss_mask * 0.44;
    let sky_amb = mix(world.ground_ambient.rgb, sky_top, 0.32 + 0.68 * t) * 0.50;
    return albedo * (world.sun_color.rgb * ndl * shadow * wave_band + sky_amb * ao + dyn_light) + sss;
}

@vertex
fn vs_grass_near(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> GrassNearOut {
    let idx = min(iid, arrayLength(&vis_blades) - 1u);
    let rec = vis_blades[idx];
    let root = vec3<f32>(rec.x, rec.y, rec.z);
    let hw = unpack2x16float(rec.hw);
    let height = hw.x;
    let base_width = hw.y;

    let bend_xz = unpack2x16float(rec.bend_misc.x);
    let right_xz = unpack2x16snorm(rec.bend_misc.y);
    let dyn_dr = unpack4x8unorm(rec.bend_misc.z);
    let droop = dyn_dr.w;
    let dyn_light = dyn_dr.xyz * dyn_dr.xyz * 16.0;

    let meta_u = rec.bend_misc.w;
    let patch_n = f32(meta_u & 127u) * (1.0 / 127.0);
    let seed_val = f32((meta_u >> 7u) & 127u) * (1.0 / 127.0);
    let shadow = f32((meta_u >> 14u) & 127u) * (1.0 / 127.0);
    let wind_mag = f32((meta_u >> 21u) & 255u) * (2.4 / 255.0);
    let flower_code = f32((meta_u >> 29u) & 7u);
    let is_flower = flower_code > 0.5;

    let blade_right = vec3<f32>(right_xz.x, 0.0, right_xz.y);
    let is_tip = vid == 4u;
    let is_mid = vid == 2u || vid == 3u;
    let mid_t = select(0.50, 0.72, is_flower);
    let t = select(select(0.0, mid_t, is_mid), 1.0, is_tip);
    let side = select(-1.0, 1.0, (vid & 1u) == 1u);
    let mid_w = select(0.58, 0.95, is_flower);
    let width_profile = select(select(1.0, mid_w, is_mid), 0.0, is_tip);

    let bend_curve = t * t;
    let spine = vec3<f32>(
        bend_xz.x * height * bend_curve,
        height * (t - droop * 0.46 * bend_curve),
        bend_xz.y * height * bend_curve,
    );
    let world_pos = root + spine + blade_right * (side * base_width * width_profile);

    let tang = normalize(vec3<f32>(
        bend_xz.x * 2.0 * t,
        max(1.0 - droop * 0.86 * t, 0.15),
        bend_xz.y * 2.0 * t,
    ));
    let face_n = normalize(cross(blade_right, tang));
    let to_cam = world.camera_pos.xyz - world_pos;
    let dist = length(to_cam);
    let view_dir = to_cam / max(dist, 0.001);
    var n = normalize(face_n + blade_right * (side * 0.34) + vec3<f32>(0.0, 0.25, 0.0));
    if dot(n, view_dir) < 0.0 {
        n = -n;
    }

    let sun_dir = world.sun_dir.xyz;
    let root_col = mix(vec3<f32>(0.008, 0.058, 0.020), vec3<f32>(0.018, 0.092, 0.026), patch_n);
    let mid_col = mix(vec3<f32>(0.018, 0.242, 0.058), vec3<f32>(0.062, 0.320, 0.054), patch_n);
    let tip_col = mix(vec3<f32>(0.095, 0.44, 0.085), vec3<f32>(0.36, 0.48, 0.088), smoothstep(0.38, 0.80, patch_n + seed_val * 0.25));
    var albedo = mix(root_col, mid_col, smoothstep(0.0, 0.54, t));
    albedo = mix(albedo, tip_col, smoothstep(0.54, 1.0, t));

    let gust_t = smoothstep(0.28, 1.45, wind_mag);
    let bs = clamp(dot(view_dir, -sun_dir), 0.0, 1.0);
    let back_scatter = bs * bs * bs;
    let sss_tint = mix(vec3<f32>(0.36, 0.76, 0.14), vec3<f32>(0.88, 0.68, 0.18), smoothstep(0.2, 1.4, world.camera_right.w));
    let sky_top = world.sky_horizon.rgb * 0.48 + world.sky_zenith.rgb * 0.52;
    let dyn_v = select(vec3<f32>(0.0), dyn_light, vid >= 2u) * (0.26 + 0.74 * t);

    var lit = eval_grass_light(t, albedo, n, sun_dir, shadow, gust_t, back_scatter, sss_tint, sky_top, dyn_v);

    let dew_fade = 1.0 - smoothstep(14.0, 22.0, dist);
    if dew_fade > 0.01 && t > 0.01 {
        let half_v = normalize(sun_dir + view_dir);
        let ndh = max(dot(n, half_v), 0.0);
        let ndh2 = ndh * ndh;
        let ndh4 = ndh2 * ndh2;
        let ndh8 = ndh4 * ndh4;
        let ndh16 = ndh8 * ndh8;
        lit += world.sun_color.rgb * (ndh16 * ndh8 * ndh4 * t * shadow * 0.16 * dew_fade);
    }

    let fog = aerial_fog_factor(world_pos, world);
    lit = mix(lit, fog.rgb, fog.w);

    var flower_lit = vec3<f32>(0.0);
    if is_flower {
        let f_col = flower_albedo(flower_code);
        let f_base = eval_grass_light(0.85, f_col, n, sun_dir, shadow, gust_t, back_scatter, sss_tint, sky_top, dyn_light * 0.89);
        flower_lit = mix(f_base, fog.rgb, fog.w);
    }

    var out: GrassNearOut;
    out.clip_pos = view_clip(world_pos);
    out.lit_t = vec4<f32>(lit, t);
    out.flower_col_flag = vec4<f32>(flower_lit, select(0.0, 1.0, is_flower));
    return out;
}

@fragment
fn fs_grass_near(in: GrassNearOut) -> @location(0) vec4<f32> {
    var col = in.lit_t.rgb;
    if in.flower_col_flag.w > 0.5 && in.lit_t.w > 0.68 {
        col = mix(col, in.flower_col_flag.rgb, smoothstep(0.68, 0.82, in.lit_t.w));
    }
    return vec4<f32>(col, 1.0);
}

@vertex
fn vs_grass_far(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> GrassFarOut {
    let idx = min(iid, arrayLength(&vis_blades) - 1u);
    let rec = vis_blades[idx];
    let root = vec3<f32>(rec.x, rec.y, rec.z);
    let hw = unpack2x16float(rec.hw);
    let height = hw.x;
    let base_width = hw.y;

    let bend_xz = unpack2x16float(rec.bend_misc.x);
    let right_xz = unpack2x16snorm(rec.bend_misc.y);
    let dyn_dr = unpack4x8unorm(rec.bend_misc.z);
    let droop = dyn_dr.w;
    let dyn_light = dyn_dr.xyz * dyn_dr.xyz * 16.0;

    let meta_u = rec.bend_misc.w;
    let patch_n = f32(meta_u & 127u) * (1.0 / 127.0);
    let seed_val = f32((meta_u >> 7u) & 127u) * (1.0 / 127.0);
    let shadow = f32((meta_u >> 14u) & 127u) * (1.0 / 127.0);
    let wind_mag = f32((meta_u >> 21u) & 255u) * (2.4 / 255.0);

    let blade_right = vec3<f32>(right_xz.x, 0.0, right_xz.y);
    let is_tip = vid == 2u;
    let t = select(0.0, 1.0, is_tip);
    let side = select(-1.0, 1.0, vid == 1u);
    let width_profile = select(1.0, 0.0, is_tip);

    let spine = vec3<f32>(
        bend_xz.x * height * t,
        height * (t - droop * 0.46 * t),
        bend_xz.y * height * t,
    );
    let world_pos = root + spine + blade_right * (side * base_width * width_profile);

    let view_dir = normalize(world.camera_pos.xyz - root);
    let sun_dir = world.sun_dir.xyz;

    let tang_mid = normalize(vec3<f32>(
        bend_xz.x * 0.9,
        max(1.0 - droop * 0.43, 0.15),
        bend_xz.y * 0.9,
    ));
    var n_base = normalize(cross(blade_right, tang_mid) + vec3<f32>(0.0, 0.28, 0.0));
    if dot(n_base, view_dir) < 0.0 {
        n_base = -n_base;
    }

    let root_col = mix(vec3<f32>(0.008, 0.058, 0.020), vec3<f32>(0.018, 0.092, 0.026), patch_n);
    let mid_col = mix(vec3<f32>(0.018, 0.242, 0.058), vec3<f32>(0.062, 0.320, 0.054), patch_n);
    let tip_col = mix(vec3<f32>(0.095, 0.44, 0.085), vec3<f32>(0.36, 0.48, 0.088), smoothstep(0.38, 0.80, patch_n + seed_val * 0.25));

    let gust_t = smoothstep(0.28, 1.45, wind_mag);
    let bs = clamp(dot(view_dir, -sun_dir), 0.0, 1.0);
    let back_scatter = bs * bs * bs;
    let sss_tint = mix(vec3<f32>(0.36, 0.76, 0.14), vec3<f32>(0.88, 0.68, 0.18), smoothstep(0.2, 1.4, world.camera_right.w));
    let sky_top = world.sky_horizon.rgb * 0.48 + world.sky_zenith.rgb * 0.52;
    let fog = aerial_fog_factor(root + vec3<f32>(0.0, height * 0.5, 0.0), world);

    let c0 = mix(
        eval_grass_light(0.0, root_col, n_base, sun_dir, shadow, gust_t, back_scatter, sss_tint, sky_top, vec3<f32>(0.0)),
        fog.rgb,
        fog.w,
    );
    let cm = mix(
        eval_grass_light(0.5, mid_col, n_base, sun_dir, shadow, gust_t, back_scatter, sss_tint, sky_top, dyn_light * 0.315),
        fog.rgb,
        fog.w,
    );
    let c1 = mix(
        eval_grass_light(1.0, tip_col, n_base, sun_dir, shadow, gust_t, back_scatter, sss_tint, sky_top, dyn_light),
        fog.rgb,
        fog.w,
    );

    var out: GrassFarOut;
    out.clip_pos = view_clip(world_pos);
    out.lit_t = vec4<f32>(select(c0, c1, is_tip), t);
    out.mid_corr = 4.0 * (cm - 0.5 * (c0 + c1));
    return out;
}

@fragment
fn fs_grass_far(in: GrassFarOut) -> @location(0) vec4<f32> {
    let t = in.lit_t.w;
    return vec4<f32>(in.lit_t.rgb + in.mid_corr * (t * (1.0 - t)), 1.0);
}
