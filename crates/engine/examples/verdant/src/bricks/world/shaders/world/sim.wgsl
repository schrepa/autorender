import package::generated::world::WorldUniforms;
import package::generated::world::GrassBlade;
import package::generated::world::PatchBounds;
import package::generated::world::Particle;
import package::generated::world::VisibleParticle;
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
@group(2) @binding(5) var<storage, read_write> vis_particles_rw: array<VisibleParticle>;
@group(2) @binding(6) var<storage, read_write> part_args_rw: array<atomic<u32>>;
@group(2) @binding(7) var<storage, read_write> patch_bounds_rw: array<PatchBounds>;

var<workgroup> wg_part_count: atomic<u32>;
var<workgroup> wg_part_base: u32;
var<workgroup> wg_wake_active: atomic<u32>;
var<workgroup> wg_min: array<vec3<f32>, 256>;
var<workgroup> wg_max: array<vec3<f32>, 256>;

@compute @workgroup_size(256)
fn cs_init_grass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let idx = gid.x;
    let total = arrayLength(&blades_rw);
    var p_lo = vec3<f32>(1.0e9, 1.0e9, 1.0e9);
    var p_hi = vec3<f32>(-1.0e9, -1.0e9, -1.0e9);

    if idx < total {
        let u0 = hash_u32(idx * 5u + 11u);
        let u1 = hash_u32(idx * 5u + 23u);
        let u2 = hash_u32(idx * 5u + 37u);
        let u3 = hash_u32(idx * 5u + 53u);
        let u4 = hash_u32(idx * 5u + 71u);

        var radius = 0.0;
        var angle = 0.0;
        if idx < 999424u {
            let local = idx % 249856u;
            let cell_id = local / 256u;
            if cell_id < 588u {
                var ring = u32(sqrt(f32(cell_id / 3u) + 0.001));
                if 3u * (ring + 1u) * (ring + 1u) <= cell_id {
                    ring += 1u;
                }
                let ring_start = 3u * ring * ring;
                let n_sec = 3u * (2u * ring + 1u);
                let sec = cell_id - ring_start;
                let r0 = f32(ring) * 2.0;
                let r1 = r0 + 2.0;
                radius = sqrt(mix(r0 * r0, r1 * r1, u0));
                angle = ((f32(sec) + u1) / f32(n_sec)) * TAU;
            } else if cell_id < 852u {
                let c = cell_id - 588u;
                let ring = c / 33u;
                let sec = c % 33u;
                let r0_sq = mix(576.0, 3600.0, f32(ring) * 0.125);
                let r1_sq = mix(576.0, 3600.0, f32(ring + 1u) * 0.125);
                radius = sqrt(mix(r0_sq, r1_sq, u0));
                angle = ((f32(sec) + u1) / 33.0) * TAU;
            } else {
                let c = cell_id - 852u;
                let ring = c / 31u;
                let sec = c % 31u;
                let r0_sq = mix(2704.0, 10816.0, f32(ring) * 0.25);
                let r1_sq = mix(2704.0, 10816.0, f32(ring + 1u) * 0.25);
                radius = sqrt(mix(r0_sq, r1_sq, u0));
                angle = ((f32(sec) + u1) / 31.0) * TAU;
            }
        } else {
            radius = sqrt(u0) * 28.0;
            angle = u1 * TAU;
        }

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

        var flower_code = 0u;
        if radius < 38.0 && u4 > 0.905 && final_h >= 0.05 {
            let flower_patch = fbm2(vec2<f32>(x, z) * 0.15 + vec2<f32>(19.4, 5.2));
            if flower_patch > 0.52 {
                let pick = fract(u4 * 37.3);
                if pick < 0.36 {
                    flower_code = 1u;
                } else if pick < 0.66 {
                    flower_code = 2u;
                } else if pick < 0.88 {
                    flower_code = 3u;
                } else {
                    flower_code = 4u;
                }
            }
        }

        var packed = 0u;
        if final_h >= 0.05 {
            let h_q = clamp(u32(clamp(final_h / 3.6, 0.0, 1.0) * 511.0 + 0.5), 1u, 511u);
            let yaw_q = u32(fract(u3) * 127.0 + 0.5) & 127u;
            let patch_q = u32(clamp(patch_n, 0.0, 1.0) * 63.0 + 0.5) & 63u;
            let seed_q = u32(clamp(u4, 0.0, 1.0) * 127.0 + 0.5) & 127u;
            packed = h_q | (yaw_q << 9u) | (patch_q << 16u) | (seed_q << 22u) | ((flower_code & 7u) << 29u);
            p_lo = vec3<f32>(x, y - 0.04, z);
            p_hi = vec3<f32>(x, y + final_h, z);
        }
        blades_rw[idx] = GrassBlade(x, y - 0.04, z, packed);
    }

    wg_min[lid] = p_lo;
    wg_max[lid] = p_hi;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if lid < s {
            wg_min[lid] = min(wg_min[lid], wg_min[lid + s]);
            wg_max[lid] = max(wg_max[lid], wg_max[lid + s]);
        }
        workgroupBarrier();
    }
    if lid == 0u && wid.x < arrayLength(&patch_bounds_rw) {
        let lo = wg_min[0];
        let hi = wg_max[0];
        if lo.x > hi.x {
            patch_bounds_rw[wid.x].sphere = vec4<f32>(0.0, 0.0, 0.0, -1.0);
        } else {
            let center = (lo + hi) * 0.5;
            let rad = length(hi - center) + 2.4;
            patch_bounds_rw[wid.x].sphere = vec4<f32>(center, rad);
        }
    }
}

@compute @workgroup_size(16, 16)
fn cs_wake_sim(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    if lid == 0u {
        atomicStore(&wg_wake_active, 0u);
    }
    workgroupBarrier();

    if gid.x < WAKE_SIZE && gid.y < WAKE_SIZE {
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
        if abs(cell.x) + abs(cell.y) + abs(cell.z) > 0.002 {
            atomicStore(&wg_wake_active, 1u);
        }
    }

    workgroupBarrier();
    if lid == 0u && wid.x < 16u && wid.y < 16u {
        let tile_idx = 65536u + wid.y * 16u + wid.x;
        if tile_idx < arrayLength(&wake_rw) {
            wake_rw[tile_idx].bend = vec4<f32>(f32(atomicLoad(&wg_wake_active)), 0.0, 0.0, 0.0);
        }
    }
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
fn cs_sim_particles(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    if lid == 0u {
        atomicStore(&wg_part_count, 0u);
    }
    workgroupBarrier();

    let idx = gid.x;
    let total = arrayLength(&particles_rw);
    var keep = false;
    var local_slot = 0u;
    var vis_out = VisibleParticle(vec4<f32>(0.0), vec4<u32>(0u));

    if idx < total {
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

        let p = particles_rw[idx].pos_phase;
        let v = particles_rw[idx].vel_kind;
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
            let pulse_l = pow(max(sin(t * (1.25 + fract(phase) * 1.1) + phase * 5.3), 0.0), 1.9);
            let warm_mix = fract(phase * 3.7);
            let col_l = mix(vec3<f32>(0.64, 1.0, 0.20), vec3<f32>(1.0, 0.72, 0.16), warm_mix);
            lights_rw[5u + idx].pos_radius = vec4<f32>(next_pos, 4.8);
            lights_rw[5u + idx].color_intensity = vec4<f32>(col_l, firefly_p * (0.28 + 1.65 * pulse_l));
        }

        let to_cam = next_pos - world.camera_pos.xyz;
        let dist_sq = dot(to_cam, to_cam);
        let max_d = select(46.0, 34.0, is_petal);
        if dist_sq >= 1.8225 && dist_sq <= max_d * max_d {
            let clip = world.view_proj * vec4<f32>(next_pos, 1.0);
            let margin = clip.w * 1.08 + 0.6;
            if clip.w > 0.2 && abs(clip.x) <= margin && abs(clip.y) <= margin {
                let dist = sqrt(dist_sq);
                if !is_petal {
                    if firefly_p >= 0.05 {
                        let is_hero = idx < 28u;
                        let wave = sin(t * (1.25 + fract(v.w) * 1.15) + phase * 5.3);
                        let w_pos = max(wave, 0.0);
                        let blink_exp = select(6.2, 1.8, is_hero);
                        let pulse = select(0.0, pow(w_pos, blink_exp), w_pos > 0.01);
                        let gate = select(0.11, 0.02, is_hero);
                        if pulse >= gate {
                            let near_fade = smoothstep(1.35, 3.8, dist);
                            let hero_boost = select(1.0, 1.65, is_hero);
                            let size = (0.022 + 0.028 * pulse) * hero_boost * (1.0 + smoothstep(14.0, 42.0, dist) * 0.55);
                            let tint = mix(vec3<f32>(0.68, 1.0, 0.20), vec3<f32>(1.0, 0.72, 0.16), fract(phase * 3.1));
                            let alpha = pulse * min(firefly_p, 1.0) * near_fade * (1.0 - smoothstep(32.0, 46.0, dist));
                            let col = vec4<f32>(tint * (2.6 + 2.8 * firefly_p * pulse), alpha);
                            vis_out = VisibleParticle(
                                vec4<f32>(next_pos, size),
                                vec4<u32>(
                                    pack2x16float(col.rg),
                                    pack2x16float(col.ba),
                                    pack2x16snorm(vec2<f32>(1.0, 0.0)),
                                    0u,
                                ),
                            );
                            keep = true;
                        }
                    }
                } else {
                    let petal_fade = smoothstep(2.2, 5.2, dist);
                    if petal_fade >= 0.02 && ((idx & 3u) == 0u || dist <= 18.0) {
                        let spin = t * (2.6 + fract(v.w) * 2.2) + phase * 5.0;
                        let tilt = 0.36 + 0.64 * abs(cos(spin * 0.7));
                        let size = (0.026 + 0.018 * fract(phase * 5.1)) * (1.0 + smoothstep(12.0, 32.0, dist) * 0.35);
                        let day_lit = mix(vec3<f32>(0.98, 0.58, 0.68), vec3<f32>(1.0, 0.84, 0.84), fract(phase * 2.3));
                        let sun_boost = world.sun_color.rgb * 0.34 + world.sky_horizon.rgb * 0.40 + vec3<f32>(0.12, 0.10, 0.14);
                        let fade = petal_fade * (1.0 - smoothstep(22.0, 34.0, dist)) * tilt;
                        let col = vec4<f32>(day_lit * sun_boost, 0.82 * fade);
                        let spin_ang = t * 2.4 + phase * 6.0;
                        vis_out = VisibleParticle(
                            vec4<f32>(next_pos, size),
                            vec4<u32>(
                                pack2x16float(col.rg),
                                pack2x16float(col.ba),
                                pack2x16snorm(vec2<f32>(cos(spin_ang), sin(spin_ang))),
                                1u,
                            ),
                        );
                        keep = true;
                    }
                }
            }
        }

        if keep {
            local_slot = atomicAdd(&wg_part_count, 1u);
        }
    }

    workgroupBarrier();
    if lid == 0u {
        let cnt = atomicLoad(&wg_part_count);
        if cnt > 0u {
            wg_part_base = atomicAdd(&part_args_rw[1], cnt);
        }
    }
    workgroupBarrier();

    if keep {
        let dst = wg_part_base + local_slot;
        if dst < arrayLength(&vis_particles_rw) {
            vis_particles_rw[dst] = vis_out;
        }
    }
}
