import package::generated::world::WorldUniforms;
import package::generated::world::PointLight;

const TAU: f32 = 6.2831853;
const WAKE_EXTENT: f32 = 42.0;
const WAKE_SIZE: u32 = 256u;

fn hash21(p_in: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p_in.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash31(p_in: vec3<f32>) -> f32 {
    var p3 = fract(p_in * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash_u32(n_in: u32) -> f32 {
    var n = (n_in ^ 61u) ^ (n_in >> 16u);
    n = n * 9u;
    n = n ^ (n >> 4u);
    n = n * 668265261u;
    n = n ^ (n >> 15u);
    return f32(n) * 2.3283064e-10;
}

fn vnoise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm2(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var val = 0.0;
    var amp = 0.5;
    for (var i = 0; i < 4; i = i + 1) {
        val += amp * vnoise2(p);
        p = mat2x2<f32>(1.62, 1.18, -1.18, 1.62) * p + vec2<f32>(17.3, 9.2);
        amp *= 0.5;
    }
    return val;
}

fn fbm2_3(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var val = 0.0;
    var amp = 0.5;
    for (var i = 0; i < 3; i = i + 1) {
        val += amp * vnoise2(p);
        p = mat2x2<f32>(1.62, 1.18, -1.18, 1.62) * p + vec2<f32>(17.3, 9.2);
        amp *= 0.5;
    }
    return val * 1.142857;
}

fn fbm2_2(p_in: vec2<f32>) -> f32 {
    let n0 = vnoise2(p_in);
    let p1 = mat2x2<f32>(1.62, 1.18, -1.18, 1.62) * p_in + vec2<f32>(17.3, 9.2);
    let n1 = vnoise2(p1);
    return (0.5 * n0 + 0.25 * n1) * 1.333333;
}

fn vnoise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));
    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);
    return mix(mix(nx00, nx10, u.y), mix(nx01, nx11, u.y), u.z);
}

fn path_center_x(z: f32) -> f32 {
    let fade = smoothstep(5.8, 16.5, z);
    let curve = -1.75 * sin((z - 5.8) * 0.135) - 0.42 * sin((z - 5.8) * 0.27);
    return curve * fade;
}

fn path_half_width(z: f32) -> f32 {
    if z < -0.6 {
        return 4.9;
    }
    if z < 6.8 {
        return 2.15;
    }
    return 1.18 + 0.16 * sin(z * 0.4);
}

fn path_clearance(x: f32, z: f32) -> f32 {
    if z >= -9.9 && z <= 0.0 {
        let dx = (abs(x) - 4.9) / 1.3;
        let dz = (abs(z + 4.9) - 4.8) / 1.3;
        return 1.0 - smoothstep(0.0, 1.0, max(dx, dz));
    }
    if z > 0.0 && z <= 23.5 {
        let cx = path_center_x(z);
        let half = path_half_width(z);
        let dist = abs(x - cx);
        let end_fade = 1.0 - smoothstep(19.5, 23.5, z);
        return (1.0 - smoothstep(half * 0.72, half + 0.65, dist)) * end_fade;
    }
    return 0.0;
}

fn crag_profile(x: f32, z: f32, phase: f32) -> f32 {
    let r1 = 1.0 - abs(sin(x * 0.032 + phase + z * 0.008));
    let r2 = 1.0 - abs(sin(x * 0.074 - phase * 1.7 + z * 0.015));
    let r3 = 1.0 - abs(cos(x * 0.155 + phase * 2.3 - z * 0.022));
    return 0.26 + 0.44 * r1 * r1 + 0.22 * r2 * r2 + 0.14 * r3 * r3;
}

fn terrain_height(x: f32, z: f32) -> f32 {
    let hx = x * 0.044;
    let hz = (z + 1.8) * 0.040;
    let main_hill = 4.95 * exp(-(hx * hx + hz * hz));

    let roll_a = 0.62 * sin(x * 0.095 + 0.35) * cos(z * 0.085 - 0.25);
    let roll_b = 0.30 * sin(x * 0.19 - z * 0.16 + 0.8);
    let roll_c = 0.14 * cos(x * 0.37 + z * 0.31);
    let flank_left = 3.2 * exp(-((x + 26.0) * (x + 26.0) * 0.0022 + (z + 10.0) * (z + 10.0) * 0.0018));
    let flank_right = 2.8 * exp(-((x - 28.0) * (x - 28.0) * 0.0020 + (z + 14.0) * (z + 14.0) * 0.0016));

    let valley_t = smoothstep(-13.0, -72.0, z);
    let valley_drop = -11.2 * valley_t;

    let front_t = smoothstep(26.0, 88.0, z);
    let front_drop = -5.5 * front_t;

    var far_mtns = 0.0;
    let far_mtn_mask = smoothstep(-76.0, -142.0, z);
    if far_mtn_mask > 0.0 {
        let bay_saddle = 1.0 - 0.68 * exp(-((x - 8.0) * (x - 8.0) * 0.00024));
        let near_crags = 26.0
            * bay_saddle
            * exp(-((z + 152.0) * (z + 152.0) * 0.00048))
            * crag_profile(x, z, 0.7);
        let mid_crags = 42.0
            * (0.38 + 0.62 * bay_saddle)
            * exp(-((z + 225.0) * (z + 225.0) * 0.00026))
            * crag_profile(x * 0.85, z, 2.1);
        let far_alps = 58.0
            * (0.45 + 0.55 * bay_saddle)
            * exp(-((z + 300.0) * (z + 300.0) * 0.00018))
            * crag_profile(x * 0.72, z, 4.3);
        let left_summit = 42.0
            * exp(-((x + 118.0) * (x + 118.0) * 0.00036 + (z + 250.0) * (z + 250.0) * 0.00028))
            * (0.72 + 0.34 * crag_profile(x * 1.4, z, 1.3));
        let right_summit = 38.0
            * exp(-((x - 115.0) * (x - 115.0) * 0.00034 + (z + 235.0) * (z + 235.0) * 0.00026))
            * (0.70 + 0.36 * crag_profile(x * 1.35, z, 3.4));
        far_mtns = (near_crags + mid_crags + far_alps + left_summit + right_summit) * far_mtn_mask;
    }

    var side_peaks = 0.0;
    let abs_x = abs(x);
    if abs_x > 72.0 {
        side_peaks = 20.0 * smoothstep(72.0, 175.0, abs_x) * crag_profile(z * 0.9, x, 0.5);
    }

    let raw = main_hill
        + (roll_a + roll_b + roll_c) * (1.0 - 0.55 * valley_t)
        + flank_left
        + flank_right
        + valley_drop
        + front_drop
        + far_mtns
        + side_peaks;

    let plateau_dx = x * 0.145;
    let plateau_dz = (z + 4.8) * 0.155;
    let plateau_w = exp(-(plateau_dx * plateau_dx + plateau_dz * plateau_dz));
    let terrace_blend = smoothstep(0.32, 0.78, plateau_w);
    let flat_top = raw * (1.0 - terrace_blend) + 4.82 * terrace_blend;

    let step_zone = smoothstep(-1.4, -0.3, z)
        * (1.0 - smoothstep(6.2, 7.6, z))
        * (1.0 - smoothstep(1.95, 3.15, abs_x));
    let step_t = clamp((z - (-1.1)) / (6.45 - (-1.1)), 0.0, 1.0);
    let step_bed = 4.62 - step_t * 1.68;
    return flat_top * (1.0 - step_zone * 0.96) + step_bed * (step_zone * 0.96);
}

fn sample_wind(xz: vec2<f32>, t: f32, strength: f32) -> vec2<f32> {
    let dir = vec2<f32>(0.860163, 0.510019);
    let perp = vec2<f32>(-0.510019, 0.860163);
    let proj = dot(xz, dir);
    let cross_p = dot(xz, perp);
    let wave_a = sin(proj * 0.20 - t * (1.95 + strength * 0.85) + sin(cross_p * 0.12 + t * 0.45) * 1.45);
    let wave_b = sin(proj * 0.45 + cross_p * 0.24 - t * (3.3 + strength * 1.2));
    let gust = 0.50 + 0.50 * wave_a;
    let ripple = 0.25 * wave_b;
    let mag = (0.30 + gust * 1.12 + ripple) * strength;
    let side = sin(proj * 0.25 - t * 2.1 + cross_p * 0.18) * 0.28 * strength;
    return dir * mag + perp * side;
}

fn canopy_shadow(p: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    let s_xz = sun_dir.xz / max(sun_dir.y, 0.26);
    let hit_sakura = p.xz + s_xz * max(9.8 - p.y, 0.0);
    let d_sakura = length(hit_sakura - vec2<f32>(-5.2, -0.8)) / 5.4;
    let sh_sakura = mix(0.30, 1.0, smoothstep(0.22, 1.06, d_sakura));

    let hit_comp = p.xz + s_xz * max(8.8 - p.y, 0.0);
    let d_comp = length(hit_comp - vec2<f32>(7.2, -5.8)) / 4.0;
    let sh_comp = mix(0.40, 1.0, smoothstep(0.25, 1.05, d_comp));

    let hit_torii = p.xz + s_xz * max(8.8 - p.y, 0.0);
    let d_tx = abs(hit_torii.x) / 3.9;
    let d_tz = abs(hit_torii.y - 2.6) / 0.85;
    let sh_torii = mix(0.48, 1.0, smoothstep(0.65, 1.25, max(d_tx, d_tz)));

    return min(min(sh_sakura, sh_comp), sh_torii);
}

fn aerial_fog_factor(world_pos: vec3<f32>, u: WorldUniforms) -> vec4<f32> {
    let delta = world_pos - u.camera_pos.xyz;
    let dist_sq = dot(delta, delta);
    if dist_sq <= 484.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let dist = sqrt(dist_sq);
    let fog_dist = dist - 22.0;
    let view_dir = delta / dist;
    let height_fade = exp(-max(world_pos.y + 4.0, 0.0) * 0.026);
    let density = mix(0.0016, 0.0027, smoothstep(0.4, 2.6, u.camera_right.w));
    let fog_t = clamp(1.0 - exp(-fog_dist * density * (0.48 + 0.56 * height_fade)), 0.0, 0.84);
    let sun_dot = max(dot(view_dir, u.sun_dir.xyz), 0.0);
    let s2 = sun_dot * sun_dot;
    let s4 = s2 * s2;
    let mie = (s4 * s2 * sun_dot) * (1.0 - u.camera_up.w * 0.80);
    let fog_col = mix(u.haze_color.rgb, u.sky_horizon.rgb, 0.24) + u.sun_color.rgb * mie * 0.16;
    return vec4<f32>(fog_col, fog_t);
}

fn apply_aerial_fog(
    color: vec3<f32>,
    world_pos: vec3<f32>,
    u: WorldUniforms,
) -> vec3<f32> {
    let fog = aerial_fog_factor(world_pos, u);
    return mix(color, fog.rgb, fog.w);
}

fn sky_radiance(view_dir: vec3<f32>, u: WorldUniforms) -> vec3<f32> {
    let up = clamp(view_dir.y, -0.15, 1.0);
    let h_t = pow(1.0 - clamp(up, 0.0, 1.0), 3.4);
    var col = mix(u.sky_zenith.rgb, u.sky_horizon.rgb, h_t);
    let mb = (up - 0.10) * 5.0;
    let mid_band = exp(-(mb * mb)) * (1.0 - u.camera_up.w * 0.75);
    col = mix(col, mix(u.haze_color.rgb, u.sky_horizon.rgb, 0.55), mid_band * 0.30);
    let below = smoothstep(0.02, -0.18, view_dir.y);
    col = mix(col, u.haze_color.rgb * 0.42, below);

    let sun_dot = max(dot(view_dir, u.sun_dir.xyz), 0.0);
    let s2 = sun_dot * sun_dot;
    let s4 = s2 * s2;
    let s8 = s4 * s4;
    let s16 = s8 * s8;
    let mie_wide = s4 * sqrt(sun_dot) * 0.11;
    let mie_tight = s16 * s16 * s16 * 0.52;
    let corona = select(0.0, pow(sun_dot, 380.0) * 1.65, sun_dot > 0.92);
    let horizon_glow = exp(-abs(view_dir.y) * 12.0) * (s2 * sqrt(sun_dot)) * 0.42;
    let day_scale = 0.25 + 0.75 * (1.0 - u.camera_up.w * 0.72);
    col += u.sun_color.rgb * (mie_wide + mie_tight + corona + horizon_glow) * day_scale;
    return col;
}
