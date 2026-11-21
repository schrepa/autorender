import package::generated::world::WorldUniforms;
import package::generated::world::GrassBlade;
import package::generated::world::Particle;
import package::generated::world::PointLight;
import package::generated::world::WakeCell;
import package::world::common::TAU;
import package::world::common::WAKE_EXTENT;
import package::world::common::WAKE_SIZE;
import package::world::common::hash_u32;
import package::world::common::fbm2;
import package::world::common::path_clearance;
import package::world::common::terrain_height;
import package::world::common::sample_wind;

@group(2) @binding(0) var<uniform> world: WorldUniforms;
@group(2) @binding(1) var<storage, read_write> blades_rw: array<GrassBlade>;
@group(2) @binding(2) var<storage, read_write> wake_rw: array<WakeCell>;
@group(2) @binding(3) var<storage, read_write> particles_rw: array<Particle>;
@group(2) @binding(4) var<storage, read_write> lights_rw: array<PointLight>;

@compute @workgroup_size(256)
fn cs_init_grass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = arrayLength(&blades_rw);
    if idx >= total {
        return;
    }

    let u0 = hash_u32(idx * 5u + 11u);
    let u1 = hash_u32(idx * 5u + 23u);
    let u2 = hash_u32(idx * 5u + 37u);
    let u3 = hash_u32(idx * 5u + 53u);
    let u4 = hash_u32(idx * 5u + 71u);

    let tier = f32(idx) / f32(total);
    var radius = 0.0;
    if tier < 0.62 {
        radius = sqrt(u0) * 28.0;
    } else if tier < 0.88 {
        radius = 24.0 + sqrt(u0) * 36.0;
    } else {
        radius = 52.0 + sqrt(u0) * 52.0;
    }
    let angle = u1 * TAU;
    var x = cos(angle) * radius;
    var z = sin(angle) * radius + 5.0;

    let clear = path_clearance(x, z);
    var height_scale = 0.34 + u2 * 0.44;
    if clear > 0.50 {
        let push_side = select(-1.0, 1.0, x >= 0.0);
        if z < 0.2 {
            x = push_side * (5.2 + u3 * 4.5);
        } else {
            x += push_side * (2.5 + u3 * 3.8);
        }
        height_scale *= 1.08;
    } else if clear > 0.14 {
        height_scale *= mix(1.05, 0.42, smoothstep(0.14, 0.50, clear));
    }

    if path_clearance(x, z) > 0.62 {
        height_scale = 0.0;
    }

    let y = terrain_height(x, z);
    if y < -4.4 {
        height_scale = 0.0;
    }

    let far_boost = 1.0 + smoothstep(26.0, 90.0, radius) * 0.85;
    let patch_n = fbm2(vec2<f32>(x, z) * 0.085 + vec2<f32>(3.1, 7.4));
    let final_h = height_scale * far_boost * (0.76 + 0.46 * patch_n);

    let yaw = u3 * TAU;
    let flower_patch = fbm2(vec2<f32>(x, z) * 0.15 + vec2<f32>(19.4, 5.2));
    var flower_code = 0.0;
    if radius < 38.0 && flower_patch > 0.52 && u4 > 0.905 {
        let pick = fract(u4 * 37.3);
        if pick < 0.36 {
            flower_code = 1.0;
        } else if pick < 0.66 {
            flower_code = 2.0;
        } else if pick < 0.88 {
            flower_code = 3.0;
        } else {
            flower_code = 4.0;
        }
    }

    let packed_hw = clamp(final_h, 0.0, 3.5) + floor(yaw * 60.0) * 4.0;
    let packed_meta = patch_n + flower_code * 2.0 + floor(u4 * 100.0) * 16.0;
    blades_rw[idx].data = vec4<f32>(x, z, packed_hw, packed_meta);
}

@compute @workgroup_size(16, 16)
fn cs_wake_sim(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= WAKE_SIZE || gid.y >= WAKE_SIZE {
        return;
    }
    let idx = gid.y * WAKE_SIZE + gid.x;
    let uv = (vec2<f32>(gid.xy) + 0.5) / f32(WAKE_SIZE);
    let world_xz = (uv * 2.0 - 1.0) * WAKE_EXTENT;
    let dt = clamp(world.time_wind.y, 0.001, 0.05);

    var cell = wake_rw[idx].bend;
    let decay = exp(-dt * 1.05);
    cell = vec4<f32>(cell.xy * decay, cell.z * exp(-dt * 0.85), 0.0);

    let to_body = world_xz - world.wake_body.xy;
    let body_d = length(to_body);
    if body_d < 2.8 {
        let falloff = 1.0 - smoothstep(0.25, 2.8, body_d);
        let push_dir = select(vec2<f32>(1.0, 0.0), to_body / body_d, body_d > 0.02);
        let speed_boost = 0.78 + min(length(world.wake_body.zw) * 0.26, 1.35);
        cell.x = clamp(cell.x + push_dir.x * falloff * speed_boost * dt * 13.0, -1.5, 1.5);
        cell.y = clamp(cell.y + push_dir.y * falloff * speed_boost * dt * 13.0, -1.5, 1.5);
        cell.z = max(cell.z, falloff * 0.96);
    }

    if world.wake_cursor.w > 0.5 {
        let to_cur = world_xz - world.wake_cursor.xy;
        let cur_d = length(to_cur);
        if cur_d < 3.2 {
            let falloff = 1.0 - smoothstep(0.15, 3.2, cur_d);
            let radial = select(vec2<f32>(0.0, 1.0), to_cur / cur_d, cur_d > 0.02);
            let swirl = vec2<f32>(radial.x * 0.78 + world.wake_cursor.z * 0.08, radial.y * 0.78);
            cell.x = clamp(cell.x + swirl.x * falloff * dt * 18.0, -1.5, 1.5);
            cell.y = clamp(cell.y + swirl.y * falloff * dt * 18.0, -1.5, 1.5);
            cell.z = max(cell.z, falloff * 0.94);
        }
    }

    wake_rw[idx].bend = cell;
}

@compute @workgroup_size(256)
fn cs_init_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = arrayLength(&particles_rw);
    if idx >= total {
        return;
    }
    let u0 = hash_u32(idx * 7u + 101u);
    let u1 = hash_u32(idx * 7u + 113u);
    let u2 = hash_u32(idx * 7u + 127u);
    let u3 = hash_u32(idx * 7u + 149u);

    let is_petal = idx >= 30000u;
    var x = 0.0;
    var y = 0.0;
    var z = 0.0;
    var vx = 0.0;
    var vy = 0.0;
    var vz = 0.0;
    var kind = 0.0;

    if !is_petal {
        let is_hero = idx < 28u;
        let rad = select(3.5 + sqrt(u0) * 86.0, 2.2 + sqrt(u0) * 18.0, is_hero);
        let ang = u1 * TAU;
        x = cos(ang) * rad;
        z = sin(ang) * rad + select(2.0, 5.8, is_hero);
        let ground = terrain_height(x, z);
        y = ground + 0.45 + u2 * select(4.8, 1.45, is_hero);
        kind = 0.0 + u3 * 0.45;
    } else {
        let ang = u0 * TAU;
        let rad = 3.0 + sqrt(u1) * 78.0;
        x = cos(ang) * rad - 3.0;
        z = sin(ang) * rad + 2.0;
        let ground = terrain_height(x, z);
        y = ground + 0.5 + u2 * 12.0;
        vx = 0.6;
        vy = -0.32 - u3 * 0.40;
        vz = 0.3;
        kind = 1.0 + u3 * 0.45;
    }

    particles_rw[idx].pos_phase = vec4<f32>(x, y, z, u3 * TAU);
    particles_rw[idx].vel_kind = vec4<f32>(vx, vy, vz, kind);
}

fn write_lantern_light(slot: u32, pos: vec3<f32>, power: f32, t: f32, phase: f32) {
    let flicker = 0.92 + 0.08 * sin(t * 7.3 + phase) * cos(t * 11.1 + phase * 1.7);
    lights_rw[slot].pos_radius = vec4<f32>(pos, 9.5);
    lights_rw[slot].color_intensity = vec4<f32>(1.0, 0.50, 0.13, power * 1.75 * flicker);
}

@compute @workgroup_size(256)
fn cs_sim_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = arrayLength(&particles_rw);
    if idx >= total {
        return;
    }

    let t = world.time_wind.x;
    let dt = clamp(world.time_wind.y, 0.001, 0.05);
    let wind_str = world.time_wind.z;
    let lantern_p = world.sky_zenith.w;
    let firefly_p = world.sky_horizon.w;

    if idx == 0u {
        write_lantern_light(0u, vec3<f32>(-3.45, terrain_height(-3.45, 8.2) + 1.88, 8.2), lantern_p, t, 0.4);
        write_lantern_light(1u, vec3<f32>(3.85, terrain_height(3.85, 8.2) + 1.88, 8.2), lantern_p, t, 1.9);
        write_lantern_light(2u, vec3<f32>(-4.25, terrain_height(-4.25, -0.25) + 1.88, -0.25), lantern_p, t, 3.2);
        write_lantern_light(3u, vec3<f32>(4.25, terrain_height(4.25, -0.25) + 1.88, -0.25), lantern_p, t, 4.7);
        lights_rw[4u].pos_radius = vec4<f32>(0.0, 6.55, -4.45, 11.5);
        lights_rw[4u].color_intensity = vec4<f32>(1.0, 0.56, 0.16, lantern_p * 2.1);
    }

    var p = particles_rw[idx].pos_phase;
    var v = particles_rw[idx].vel_kind;
    let pos = p.xyz;
    let phase = p.w;
    let is_petal = v.w >= 0.9;
    let w_vec = sample_wind(pos.xz, t, wind_str);

    var next_pos = pos;
    var next_vel = v.xyz;

    if !is_petal {
        let is_hero = idx < 28u;
        let swirl = vec3<f32>(
            sin(t * 0.65 + phase * 1.9 + pos.z * 0.14),
            cos(t * 1.05 + phase * 2.3 + pos.x * 0.17) * 0.44,
            cos(t * 0.60 + phase * 1.5 - pos.x * 0.13),
        );
        let desired = vec3<f32>(w_vec.x * 0.22, 0.0, w_vec.y * 0.22) + swirl * 0.62;
        next_vel = mix(next_vel, desired, clamp(dt * 2.2, 0.0, 1.0));
        next_pos += next_vel * dt;

        let ground = terrain_height(next_pos.x, next_pos.z);
        let target_h = ground + 0.48 + (0.5 + 0.5 * sin(t * 0.50 + phase * 3.1)) * select(3.6, 1.35, is_hero);
        next_pos.y = mix(next_pos.y, target_h, clamp(dt * 1.4, 0.0, 1.0));

        let center_z = select(2.0, 5.8, is_hero);
        let bound_r = select(90.0, 20.0, is_hero);
        let rel_xz = next_pos.xz - vec2<f32>(0.0, center_z);
        if dot(rel_xz, rel_xz) > bound_r * bound_r {
            next_pos.x = -rel_xz.x * 0.85;
            next_pos.z = -rel_xz.y * 0.85 + center_z;
            next_pos.y = terrain_height(next_pos.x, next_pos.z) + 1.1;
        }
    } else {
        let flutter = vec3<f32>(
            sin(t * 2.4 + phase * 4.1) * 0.55,
            -0.30 - 0.16 * sin(t * 1.8 + phase * 2.7),
            cos(t * 2.1 + phase * 3.3) * 0.55,
        );
        next_vel = vec3<f32>(w_vec.x * 1.45, 0.0, w_vec.y * 1.45) + flutter;
        next_pos += next_vel * dt;
        let ground = terrain_height(next_pos.x, next_pos.z);
        if next_pos.y < ground + 0.08 || abs(next_pos.x) > 85.0 || abs(next_pos.z - 2.0) > 85.0 {
            let s0 = hash_u32(idx * 13u + u32(t * 17.0));
            let s1 = hash_u32(idx * 17u + u32(t * 23.0) + 7u);
            let s2 = hash_u32(idx * 19u + u32(t * 11.0) + 13u);
            let pick_tree = (idx % 5u) == 0u;
            if pick_tree {
                let ang = s0 * TAU;
                let r = sqrt(s1) * 5.2;
                let origin = select(vec3<f32>(7.4, 9.2, -5.6), vec3<f32>(-5.5, 9.8, -1.0), (idx % 2u) == 0u);
                next_pos = origin + vec3<f32>(cos(ang) * r, (s2 - 0.3) * 2.8, sin(ang) * r);
            } else {
                next_pos.x = -76.0 + s0 * 152.0;
                next_pos.z = -68.0 + s1 * 140.0;
                next_pos.y = terrain_height(next_pos.x, next_pos.z) + 4.5 + s2 * 10.5;
            }
        }
    }

    particles_rw[idx].pos_phase = vec4<f32>(next_pos, phase);
    particles_rw[idx].vel_kind = vec4<f32>(next_vel, v.w);

    if idx < 28u {
        let pulse = pow(max(sin(t * (1.25 + fract(phase) * 1.1) + phase * 5.3), 0.0), 1.9);
        let warm_mix = fract(phase * 3.7);
        let col = mix(vec3<f32>(0.64, 1.0, 0.20), vec3<f32>(1.0, 0.72, 0.16), warm_mix);
        lights_rw[5u + idx].pos_radius = vec4<f32>(next_pos, 4.8);
        lights_rw[5u + idx].color_intensity = vec4<f32>(col, firefly_p * (0.28 + 1.65 * pulse));
    }
}
