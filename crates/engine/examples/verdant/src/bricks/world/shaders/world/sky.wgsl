import package::generated::world::WorldUniforms;
import package::world::common::fbm2;
import package::world::common::hash21;
import package::world::common::hash31;
import package::world::common::sky_radiance;

@group(2) @binding(0) var<uniform> world: WorldUniforms;

struct SkyOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_sky(@builtin(vertex_index) idx: u32) -> SkyOut {
    let x = f32((idx << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(idx & 2u) * 2.0 - 1.0;
    var out: SkyOut;
    out.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    out.ndc = vec2<f32>(x, y);
    return out;
}

fn cloud_layer(dir: vec3<f32>, t: f32) -> vec3<f32> {
    if dir.y < 0.01 {
        return vec3<f32>(0.0);
    }
    let plane_uv = dir.xz / (dir.y + 0.14);
    let wind_drift = vec2<f32>(t * 0.015, t * 0.009) * (0.5 + world.time_wind.z * 0.5);
    let n_base = fbm2(plane_uv * 0.52 - wind_drift + vec2<f32>(2.1, -1.2));
    let n_billow = fbm2(plane_uv * 1.38 - wind_drift * 1.35 + vec2<f32>(4.2, 8.7));
    let shape = n_base * 0.70 + n_billow * 0.38;
    let density = smoothstep(0.42, 0.70, shape);
    let horizon_fade = smoothstep(0.01, 0.085, dir.y) * (1.0 - smoothstep(0.72, 0.98, dir.y) * 0.42);

    let sun_step = world.sun_dir.xz * 0.32;
    let n_lit = fbm2((plane_uv + sun_step) * 0.52 - wind_drift + vec2<f32>(2.1, -1.2));
    let lit = clamp(0.56 - (n_lit - n_base) * 3.8 + (shape - 0.50) * 0.75, 0.04, 1.0);

    let cirrus_uv = dir.xz / (dir.y + 0.24);
    let cirrus = smoothstep(0.52, 0.80, fbm2(cirrus_uv * vec2<f32>(0.90, 2.2) - wind_drift * 1.8)) * 0.36 * horizon_fade;

    return vec3<f32>(density * horizon_fade * 0.95, lit, cirrus);
}

fn star_field(dir: vec3<f32>, star_power: f32, t: f32) -> vec3<f32> {
    if star_power <= 0.01 || dir.y < 0.02 {
        return vec3<f32>(0.0);
    }
    let cell = floor(dir * 260.0);
    let rnd = hash31(cell);
    var stars = vec3<f32>(0.0);
    if rnd > 0.990 {
        let local = fract(dir * 260.0) - 0.5;
        let spot = exp(-dot(local, local) * 36.0);
        let twinkle = 0.62 + 0.38 * sin(t * (2.2 + rnd * 5.0) + rnd * 90.0);
        let tint = mix(vec3<f32>(0.72, 0.86, 1.0), vec3<f32>(1.0, 0.88, 0.68), fract(rnd * 97.3));
        stars = tint * spot * twinkle * 2.8;
    }
    let band_coord = abs(dot(dir, vec3<f32>(0.42025, 0.68041, -0.60036)));
    let milky = exp(-band_coord * 7.0) * fbm2(dir.xz * 4.2 + vec2<f32>(dir.y * 3.0)) * 0.16;
    let fade = smoothstep(0.02, 0.22, dir.y) * star_power;
    return (stars + vec3<f32>(0.34, 0.48, 0.82) * milky) * fade;
}

@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
    let far_h = world.inv_view_proj * vec4<f32>(in.ndc, 0.001, 1.0);
    let far_pos = far_h.xyz / far_h.w;
    let dir = normalize(far_pos - world.camera_pos.xyz);

    var col = sky_radiance(dir, world);
    col += star_field(dir, world.sun_dir.w, world.time_wind.x);

    let sun_dot = max(dot(dir, world.sun_dir.xyz), 0.0);
    let moon_t = world.camera_up.w;
    let disk_inner = mix(0.99915, 0.99942, moon_t);
    let disk_outer = mix(0.99855, 0.99905, moon_t);
    let disk = smoothstep(disk_outer, disk_inner, sun_dot);
    let disk_boost = mix(6.8, 4.4, moon_t);
    var disk_col = world.sun_color.rgb * disk_boost;
    if moon_t > 0.5 {
        let crater = 0.80 + 0.20 * hash21(floor(dir.xy * 180.0));
        disk_col = vec3<f32>(0.90, 0.95, 1.08) * 4.6 * crater;
    }
    col += disk_col * disk;

    let cloud = cloud_layer(dir, world.time_wind.x);
    if cloud.z > 0.001 {
        let cirrus_col = mix(world.cloud_lit.rgb, world.sky_horizon.rgb * 1.15, 0.35);
        col = mix(col, cirrus_col, cloud.z);
    }
    if cloud.x > 0.001 {
        let silver = pow(sun_dot, 12.0) * (1.15 - moon_t * 0.45);
        let c_col = mix(world.cloud_shadow.rgb, world.cloud_lit.rgb, cloud.y)
            + world.sun_color.rgb * silver * 0.36;
        col = mix(col, c_col, cloud.x);
    }

    return vec4<f32>(col, 1.0);
}
