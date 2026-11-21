import package::engine::view::view_clip;
import package::generated::world::WorldUniforms;
import package::generated::world::GrassBlade;
import package::generated::world::WakeCell;
import package::generated::world::PointLight;
import package::world::common::WAKE_EXTENT;
import package::world::common::WAKE_SIZE;
import package::world::common::terrain_height;
import package::world::common::sample_wind;
import package::world::common::canopy_shadow;
import package::world::common::apply_aerial_fog;

@group(2) @binding(0) var<uniform> world: WorldUniforms;
@group(2) @binding(1) var<storage, read> blades: array<GrassBlade>;
@group(2) @binding(2) var<storage, read> wake: array<WakeCell>;
@group(2) @binding(3) var<storage, read> lights: array<PointLight>;

struct GrassVsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) params: vec4<f32>,
    @location(3) flower_shadow: vec2<f32>,
    @location(4) dyn_light: vec3<f32>,
}

fn sample_wake(xz: vec2<f32>) -> vec3<f32> {
    let uv = xz / (WAKE_EXTENT * 2.0) + 0.5;
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
        return vec3<f32>(0.0);
    }
    let f = uv * f32(WAKE_SIZE - 1u);
    let i0 = vec2<u32>(floor(f));
    let i1 = min(i0 + vec2<u32>(1u), vec2<u32>(WAKE_SIZE - 1u));
    let frac = fract(f);
    let c00 = wake[i0.y * WAKE_SIZE + i0.x].bend.xyz;
    let c10 = wake[i0.y * WAKE_SIZE + i1.x].bend.xyz;
    let c01 = wake[i1.y * WAKE_SIZE + i0.x].bend.xyz;
    let c11 = wake[i1.y * WAKE_SIZE + i1.x].bend.xyz;
    return mix(mix(c00, c10, frac.x), mix(c01, c11, frac.x), frac.y);
}

fn grass_dynamic_lights(pos: vec3<f32>, dist_sq: f32) -> vec3<f32> {
    if world.sky_zenith.w + world.sky_horizon.w < 0.05 || dist_sq > 1400.0 {
        return vec3<f32>(0.0);
    }
    var accum = vec3<f32>(0.0);
    let max_i = select(5u, arrayLength(&lights), world.sky_horizon.w > 0.25 && dist_sq < 960.0);
    for (var i = 0u; i < max_i; i = i + 1u) {
        let pr = lights[i].pos_radius;
        let ci = lights[i].color_intensity;
        if ci.w <= 0.01 {
            continue;
        }
        let d_vec = pr.xyz - pos;
        let d2 = dot(d_vec, d_vec);
        let r2 = pr.w * pr.w;
        if d2 >= r2 {
            continue;
        }
        let d = sqrt(max(d2, 0.001));
        let atten = pow(1.0 - d / pr.w, 2.2) / (1.0 + 0.45 * max(d2, 0.45));
        accum += ci.rgb * (ci.w * atten);
    }
    return accum;
}

@vertex
fn vs_grass(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> GrassVsOut {
    var out: GrassVsOut;
    let rec = blades[iid].data;
    let base_xz = rec.xy;
    let packed_hw = rec.z;
    let height = fract(packed_hw * 0.25) * 4.0;
    if height < 0.05 {
        out.clip_pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    let ground_y = terrain_height(base_xz.x, base_xz.y);
    let root = vec3<f32>(base_xz.x, ground_y - 0.04, base_xz.y);

    let delta = root - world.camera_pos.xyz;
    let dist_sq = dot(delta, delta);
    if dist_sq > 13000.0 {
        out.clip_pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    let root_clip = view_clip(root + vec3<f32>(0.0, height * 0.5, 0.0));
    let margin = root_clip.w * 1.16 + 2.0;
    if root_clip.w < -1.5 || abs(root_clip.x) > margin || abs(root_clip.y) > margin {
        out.clip_pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    let yaw = floor(packed_hw * 0.25) / 60.0;
    let packed_meta = rec.w;
    let seed_val = floor(packed_meta / 16.0) * 0.01;
    let rem_meta = packed_meta - floor(packed_meta / 16.0) * 16.0;
    let flower_code = floor(rem_meta * 0.5);
    let patch_n = rem_meta - flower_code * 2.0;

    let is_tip = vid == 6u;
    let seg = f32(vid / 2u) / 3.0;
    let side = select(-1.0, 1.0, (vid & 1u) == 1u);
    let t = select(seg, 1.0, is_tip);

    let time = world.time_wind.x;
    let wind_str = world.time_wind.z;
    let w_vec = sample_wind(base_xz, time, wind_str);
    let flutter = sin(time * (4.2 + wind_str * 1.6) + seed_val * 25.0 + base_xz.x * 1.3) * 0.08 * (0.4 + wind_str);
    let wake_v = sample_wake(base_xz);

    let blade_dir = vec2<f32>(cos(yaw), sin(yaw));
    let arch_lean = blade_dir * (0.22 + 0.24 * seed_val);
    let bend_xz = arch_lean + w_vec * (0.42 + 0.20 * seed_val) + vec2<f32>(flutter, flutter * 0.6) + wake_v.xy * 1.35;
    let bend_len = length(bend_xz);
    let droop = min(bend_len * 0.44 + wake_v.z * 0.72, 0.82);

    let bend_curve = t * t;
    let spine = vec3<f32>(
        bend_xz.x * height * bend_curve,
        height * (t - droop * 0.46 * bend_curve),
        bend_xz.y * height * bend_curve,
    );

    let dist = sqrt(max(dist_sq, 0.001));
    let cam_right = normalize(vec3<f32>(world.camera_right.x, 0.0, world.camera_right.z) + vec3<f32>(0.0001, 0.0, 0.0));
    let blade_right = normalize(mix(vec3<f32>(-blade_dir.y, 0.0, blade_dir.x), cam_right, smoothstep(7.0, 26.0, dist) * 0.72));

    var base_width = (0.034 + 0.016 * seed_val) * (1.0 + smoothstep(14.0, 66.0, dist) * 1.45);
    var width_profile = pow(1.0 - t, 0.78);
    if flower_code > 0.5 && t > 0.62 {
        width_profile = select(0.95, 0.0, is_tip);
    }
    let half_w = base_width * width_profile * select(1.0, 0.0, is_tip);
    let world_pos = root + spine + blade_right * (side * half_w);

    let tang = normalize(vec3<f32>(
        bend_xz.x * 2.0 * t,
        max(1.0 - droop * 0.86 * t, 0.15),
        bend_xz.y * 2.0 * t,
    ));
    let face_n = normalize(cross(blade_right, tang));
    let curved_n = normalize(face_n + blade_right * (side * 0.34) + vec3<f32>(0.0, 0.25, 0.0));

    out.clip_pos = view_clip(world_pos);
    out.world_pos = world_pos;
    out.normal = curved_n;
    out.params = vec4<f32>(t, patch_n, seed_val, clamp(length(w_vec), 0.0, 2.4));
    out.flower_shadow = vec2<f32>(flower_code, canopy_shadow(world_pos, world.sun_dir.xyz));
    out.dyn_light = select(vec3<f32>(0.0), grass_dynamic_lights(world_pos, dist_sq), vid >= 2u);
    return out;
}

@fragment
fn fs_grass(in: GrassVsOut) -> @location(0) vec4<f32> {
    let t = in.params.x;
    let patch_n = in.params.y;
    let seed_val = in.params.z;
    let wind_mag = in.params.w;
    let flower_code = in.flower_shadow.x;
    let shadow = in.flower_shadow.y;

    var n = normalize(in.normal);
    let view_dir = normalize(world.camera_pos.xyz - in.world_pos);
    if dot(n, view_dir) < 0.0 {
        n = -n;
    }
    let sun_dir = world.sun_dir.xyz;

    let root_col = mix(vec3<f32>(0.008, 0.058, 0.020), vec3<f32>(0.018, 0.092, 0.026), patch_n);
    let mid_col = mix(vec3<f32>(0.018, 0.242, 0.058), vec3<f32>(0.062, 0.320, 0.054), patch_n);
    let tip_col = mix(vec3<f32>(0.095, 0.44, 0.085), vec3<f32>(0.36, 0.48, 0.088), smoothstep(0.38, 0.80, patch_n + seed_val * 0.25));

    var albedo = mix(root_col, mid_col, smoothstep(0.0, 0.54, t));
    albedo = mix(albedo, tip_col, smoothstep(0.54, 1.0, t));

    if flower_code > 0.5 && t > 0.68 {
        var f_col = vec3<f32>(0.98, 0.76, 0.14);
        if flower_code > 1.5 && flower_code < 2.5 {
            f_col = vec3<f32>(0.98, 0.94, 0.86);
        } else if flower_code > 2.5 && flower_code < 3.5 {
            f_col = vec3<f32>(0.36, 0.58, 0.98);
        } else if flower_code > 3.5 {
            f_col = vec3<f32>(0.95, 0.22, 0.16);
        }
        albedo = mix(albedo, f_col, smoothstep(0.68, 0.82, t));
    }

    let ao = mix(0.14, 1.0, pow(t, 0.80));
    let ndl = clamp((dot(n, sun_dir) + 0.36) / 1.36, 0.0, 1.0);
    let gust_t = smoothstep(0.28, 1.45, wind_mag);
    let wave_band = mix(0.62, 1.48, gust_t * (0.25 + 0.75 * t));

    let back_scatter = pow(clamp(dot(view_dir, -sun_dir), 0.0, 1.0), 3.2);
    let sss_mask = t * t * (0.16 + 0.84 * back_scatter) * shadow;
    let sss_tint = mix(vec3<f32>(0.36, 0.76, 0.14), vec3<f32>(0.88, 0.68, 0.18), smoothstep(0.2, 1.4, world.camera_right.w));
    let sss = world.sun_color.rgb * sss_tint * sss_mask * 0.44;

    let half_v = normalize(sun_dir + view_dir);
    let dew_spec = pow(max(dot(n, half_v), 0.0), 28.0) * t * shadow * 0.16 * world.sun_color.rgb;

    let sky_amb = mix(world.ground_ambient.rgb, world.sky_horizon.rgb * 0.48 + world.sky_zenith.rgb * 0.52, 0.32 + 0.68 * t) * 0.50;
    let dyn_light = in.dyn_light * (0.26 + 0.74 * t);

    var color = albedo * (world.sun_color.rgb * ndl * shadow * wave_band + sky_amb * ao + dyn_light)
        + sss
        + dew_spec;
    color = apply_aerial_fog(color, in.world_pos, world);
    return vec4<f32>(color, 1.0);
}
