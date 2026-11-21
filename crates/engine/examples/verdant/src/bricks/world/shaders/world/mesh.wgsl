import package::engine::view::view_clip;
import package::generated::world::WorldUniforms;
import package::generated::world::MeshVertex;
import package::generated::world::PointLight;
import package::world::common::fbm2;
import package::world::common::vnoise3;
import package::world::common::sample_wind;
import package::world::common::canopy_shadow;
import package::world::common::apply_aerial_fog;
import package::world::common::sky_radiance;

@group(2) @binding(0) var<uniform> world: WorldUniforms;
@group(2) @binding(1) var<storage, read> mesh_verts: array<MeshVertex>;
@group(2) @binding(2) var<storage, read> mesh_indices: array<u32>;
@group(2) @binding(3) var<storage, read> lights: array<PointLight>;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv_extra: vec4<f32>,
    @location(3) tint: vec4<f32>,
    @location(4) mat_ao: vec2<f32>,
}

@vertex
fn vs_mesh(@builtin(vertex_index) vid: u32) -> VsOut {
    let idx = mesh_indices[vid];
    let v = mesh_verts[idx];
    var pos = v.pos_mat.xyz;
    let mat_id = v.pos_mat.w;
    let t = world.time_wind.x;
    let wind_str = world.time_wind.z;

    if mat_id > 7.5 && mat_id < 9.5 {
        let w = sample_wind(pos.xz, t, wind_str);
        let phase = v.uv_extra.z;
        let gust = sin(t * 2.4 + phase + pos.x * 0.45) * 0.085 * (0.45 + wind_str);
        pos += vec3<f32>(w.x * 0.14 + gust, cos(t * 2.1 + phase) * 0.045 * wind_str, w.y * 0.14);
    } else if mat_id > 5.5 && mat_id < 6.5 && v.uv_extra.z > 0.5 {
        let w = sample_wind(pos.xz, t, wind_str);
        let sway = (v.uv_extra.z - 0.5) * 0.06 * (0.5 + wind_str);
        pos.x += (w.x * 0.35 + sin(t * 3.8 + pos.x * 2.0) * 0.4) * sway;
        pos.z += (w.y * 0.35 + cos(t * 3.1 + pos.x * 1.7) * 0.4) * sway;
    }

    var out: VsOut;
    out.clip_pos = view_clip(pos);
    out.world_pos = pos;
    out.normal = v.norm_ao.xyz;
    out.uv_extra = v.uv_extra;
    out.tint = v.tint;
    out.mat_ao = vec2<f32>(mat_id, v.norm_ao.w);
    return out;
}

fn eval_dynamic_lights(pos: vec3<f32>, n: vec3<f32>, wrap: f32) -> vec3<f32> {
    if world.sky_zenith.w + world.sky_horizon.w < 0.05 {
        return vec3<f32>(0.0);
    }
    var accum = vec3<f32>(0.0);
    let max_i = select(5u, arrayLength(&lights), world.sky_horizon.w > 0.12);
    for (var i = 0u; i < max_i; i = i + 1u) {
        let pr = lights[i].pos_radius;
        let ci = lights[i].color_intensity;
        if ci.w <= 0.005 {
            continue;
        }
        let to_l = pr.xyz - pos;
        let d2 = dot(to_l, to_l);
        let r2 = pr.w * pr.w;
        if d2 >= r2 {
            continue;
        }
        let d = sqrt(max(d2, 0.001));
        let soft_d2 = max(d2, 0.95);
        let atten = pow(1.0 - d / pr.w, 2.0) / (1.0 + 0.52 * soft_d2);
        let ndl = clamp((dot(n, to_l / d) + wrap) / (1.0 + wrap), 0.0, 1.0);
        accum += ci.rgb * (ci.w * atten * (0.14 + 0.86 * ndl));
    }
    return accum;
}

fn shade_water(world_pos: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let t = world.time_wind.x;
    let p = world_pos.xz * 0.14;
    let n1 = fbm2(p + vec2<f32>(t * 0.18, t * 0.11));
    let n2 = fbm2(p * 1.9 - vec2<f32>(t * 0.14, -t * 0.16));
    let ripple = vec3<f32>((n1 - 0.5) * 0.09, 1.0, (n2 - 0.5) * 0.09);
    let n = normalize(ripple);
    let refl = reflect(-view_dir, n);
    let sky_r = sky_radiance(vec3<f32>(refl.x, max(refl.y, 0.02), refl.z), world);
    let fresnel = 0.04 + 0.96 * pow(1.0 - max(dot(n, view_dir), 0.0), 4.5);
    let deep = mix(vec3<f32>(0.015, 0.055, 0.10), world.sky_horizon.rgb * 0.26, 0.25);
    let half_v = normalize(world.sun_dir.xyz + view_dir);
    let spec = pow(max(dot(n, half_v), 0.0), 190.0) * 3.8;
    let col = mix(deep, sky_r, fresnel) + world.sun_color.rgb * spec;
    return apply_aerial_fog(col, world_pos, world);
}

@fragment
fn fs_mesh(in: VsOut) -> @location(0) vec4<f32> {
    let mat_id = in.mat_ao.x;
    let ao = in.mat_ao.y;
    let view_dir = normalize(world.camera_pos.xyz - in.world_pos);
    let sun_dir = world.sun_dir.xyz;

    if mat_id > 0.5 && mat_id < 1.5 {
        return vec4<f32>(shade_water(in.world_pos, view_dir), 1.0);
    }

    var n = normalize(in.normal);
    var albedo = in.tint.rgb;
    var rough = 0.84;
    var metallic = 0.0;
    var emissive = vec3<f32>(0.0);

    let grain = vnoise3(in.world_pos * 3.8);
    let fine = vnoise3(in.world_pos * 14.0);
    let macro_n = fbm2(in.world_pos.xz * 0.08 + vec2<f32>(2.4, 5.1));

    if mat_id < 0.5 {
        let dist = length(in.world_pos.xz);
        let path = in.uv_extra.x;
        let meadow_col = mix(vec3<f32>(0.032, 0.095, 0.024), vec3<f32>(0.095, 0.185, 0.040), macro_n);
        let soil_col = mix(vec3<f32>(0.085, 0.072, 0.055), vec3<f32>(0.145, 0.125, 0.092), grain);
        let rock_col = vec3<f32>(0.11, 0.12, 0.12) * (0.72 + 0.46 * grain);
        let mtn_col = mix(vec3<f32>(0.030, 0.068, 0.078), vec3<f32>(0.055, 0.105, 0.092), macro_n);
        let snow_cap = smoothstep(32.0, 56.0, in.world_pos.y) * smoothstep(0.55, 0.85, n.y);
        let slope = 1.0 - smoothstep(0.62, 0.88, n.y);
        albedo = mix(meadow_col, soil_col, path * 0.72);
        albedo = mix(albedo, rock_col, slope * 0.78);
        albedo = mix(albedo, mtn_col, smoothstep(46.0, 130.0, dist));
        albedo = mix(albedo, vec3<f32>(0.62, 0.66, 0.72), snow_cap * 0.55);
        rough = 0.94;
    } else if mat_id < 2.5 {
        let chisel = (grain - 0.5) * 0.24 + (fine - 0.5) * 0.16;
        n = normalize(n + vec3<f32>(chisel, 0.0, (fine - 0.5) * 0.20));
        albedo = albedo * (0.70 + 0.42 * grain) * (0.88 + 0.22 * fine);
        let moss_mask = smoothstep(0.38, 0.80, n.y + (grain - 0.44) * 0.58) * in.uv_extra.w;
        let moss_col = mix(vec3<f32>(0.055, 0.155, 0.036), vec3<f32>(0.135, 0.265, 0.058), fine);
        albedo = mix(albedo, moss_col, moss_mask * 0.78);
        rough = mix(0.84, 0.95, moss_mask);
    } else if mat_id < 3.5 {
        let wood = vnoise3(vec3<f32>(in.world_pos.x * 8.0, in.world_pos.y * 2.2, in.world_pos.z * 8.0));
        albedo = albedo * (0.76 + 0.34 * wood);
        let base_wear = 1.0 - smoothstep(4.2, 5.4, in.world_pos.y);
        albedo = mix(albedo, albedo * 0.52 + vec3<f32>(0.04, 0.07, 0.03), base_wear * 0.45);
        rough = 0.42 + 0.24 * wood;
    } else if mat_id < 4.5 {
        let tile = sin(in.world_pos.x * 16.0) * 0.06;
        n = normalize(n + vec3<f32>(tile, 0.0, (fine - 0.5) * 0.08));
        albedo = albedo * (0.80 + 0.32 * grain);
        let moss_roof = smoothstep(0.55, 0.88, n.y) * in.uv_extra.w * smoothstep(0.46, 0.72, macro_n);
        albedo = mix(albedo, vec3<f32>(0.08, 0.17, 0.05), moss_roof * 0.52);
        rough = 0.54 + 0.22 * fine;
    } else if mat_id < 5.5 {
        let flicker = 0.92 + 0.08 * sin(world.time_wind.x * 7.5 + in.world_pos.x * 2.3);
        let power = (0.42 + world.sky_zenith.w * 2.45) * flicker;
        albedo = vec3<f32>(1.0, 0.72, 0.32);
        emissive = vec3<f32>(1.0, 0.56, 0.16) * power * 3.1;
        rough = 0.6;
    } else if mat_id < 6.5 {
        albedo = albedo * (0.84 + 0.24 * fine);
        rough = 0.90;
    } else if mat_id < 7.5 {
        let bark = vnoise3(vec3<f32>(in.world_pos.x * 9.0, in.world_pos.y * 3.0, in.world_pos.z * 9.0));
        n = normalize(n + vec3<f32>((bark - 0.5) * 0.38, 0.0, (fine - 0.5) * 0.26));
        albedo = albedo * (0.62 + 0.58 * bark);
        let moss = smoothstep(0.42, 0.84, n.y) * in.uv_extra.w * smoothstep(0.40, 0.68, grain);
        albedo = mix(albedo, vec3<f32>(0.07, 0.18, 0.05), moss * 0.72);
        rough = 0.92;
    } else if mat_id > 9.5 {
        albedo = albedo * (0.88 + 0.20 * fine);
        rough = 0.26;
        metallic = 0.88;
    }

    let shadow = canopy_shadow(in.world_pos, sun_dir);
    let ndl = max(dot(n, sun_dir), 0.0);
    let half_v = normalize(sun_dir + view_dir);
    let ndh = max(dot(n, half_v), 0.0);
    let ndv = max(dot(n, view_dir), 0.001);
    let spec_pow = mix(96.0, 12.0, rough * rough);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - ndv, 5.0);
    let spec = fresnel * (pow(ndh, spec_pow) * (spec_pow + 8.0) * 0.034) * ndl * shadow;

    let hemi_t = n.y * 0.5 + 0.5;
    let ambient = mix(world.ground_ambient.rgb, mix(world.sky_horizon.rgb * 0.65, world.sky_zenith.rgb, 0.58), hemi_t) * 0.58;
    let direct = world.sun_color.rgb * (ndl * shadow * (1.0 - metallic * 0.85));
    let dyn_light = eval_dynamic_lights(in.world_pos, n, 0.20);

    var lit = albedo * (direct + ambient * ao + dyn_light) + spec * world.sun_color.rgb + emissive;
    lit = apply_aerial_fog(lit, in.world_pos, world);
    return vec4<f32>(lit, 1.0);
}

@fragment
fn fs_foliage(in: VsOut) -> @location(0) vec4<f32> {
    let mat_id = in.mat_ao.x;
    let ao = in.mat_ao.y;
    let uv = in.uv_extra.xy * 2.0 - 1.0;

    if mat_id > 7.5 && mat_id < 9.5 {
        let ang = atan2(uv.y, uv.x);
        let lobes = select(5.0, 7.0, mat_id > 8.5);
        let ruffle = 0.72 + 0.24 * sin(ang * lobes + in.uv_extra.z) + 0.09 * cos(ang * (lobes * 2.0 - 1.0));
        let d = length(uv);
        if d > ruffle {
            discard;
        }
    }

    let n = normalize(in.normal);
    let view_dir = normalize(world.camera_pos.xyz - in.world_pos);
    let sun_dir = world.sun_dir.xyz;

    let wrap = select(0.45, 0.30, mat_id > 8.5);
    let ndl = clamp((dot(n, sun_dir) + wrap) / (1.0 + wrap), 0.0, 1.0);
    let back_dot = pow(clamp(dot(view_dir, -sun_dir), 0.0, 1.0), 2.8);
    let sss = world.sun_color.rgb * in.tint.w * (0.14 + 0.86 * back_dot) * (0.28 + 0.72 * ao);

    let hemi_t = n.y * 0.5 + 0.5;
    let ambient = mix(world.ground_ambient.rgb, world.sky_horizon.rgb * 0.58 + world.sky_zenith.rgb * 0.45, hemi_t) * 0.56;
    let dyn_light = eval_dynamic_lights(in.world_pos, n, 0.55);

    var albedo = in.tint.rgb;
    if mat_id > 7.5 && mat_id < 8.5 {
        let deep_rose = vec3<f32>(0.42, 0.11, 0.22);
        let sunlit_blush = vec3<f32>(1.0, 0.80, 0.82);
        let base_sakura = mix(deep_rose, albedo, smoothstep(0.18, 0.82, ao));
        albedo = mix(base_sakura, sunlit_blush, smoothstep(0.65, 1.0, ao * ndl) * 0.48);
    }

    var lit = albedo * (world.sun_color.rgb * ndl * 0.76 + ambient * (0.35 + 0.65 * ao) + dyn_light) + albedo * sss * 0.38;
    lit = apply_aerial_fog(lit, in.world_pos, world);
    return vec4<f32>(lit, 1.0);
}
