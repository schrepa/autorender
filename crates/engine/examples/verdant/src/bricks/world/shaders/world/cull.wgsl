import package::generated::world::WorldUniforms;
import package::generated::world::GrassBlade;
import package::generated::world::PatchBounds;
import package::generated::world::VisibleBlade;
import package::generated::world::WakeCell;
import package::generated::world::PointLight;
import package::world::common::TAU;
import package::world::common::WAKE_EXTENT;
import package::world::common::WAKE_SIZE;
import package::world::common::sample_wind;
import package::world::common::canopy_shadow;

@group(2) @binding(0) var<uniform> world: WorldUniforms;
@group(2) @binding(1) var<storage, read> blades: array<GrassBlade>;
@group(2) @binding(2) var<storage, read> patch_bounds: array<PatchBounds>;
@group(2) @binding(3) var<storage, read> wake: array<WakeCell>;
@group(2) @binding(4) var<storage, read> lights: array<PointLight>;
@group(2) @binding(5) var<storage, read_write> vis_near0_rw: array<VisibleBlade>;
@group(2) @binding(6) var<storage, read_write> vis_near1_rw: array<VisibleBlade>;
@group(2) @binding(7) var<storage, read_write> vis_far_rw: array<VisibleBlade>;
@group(2) @binding(8) var<storage, read_write> grass_args_rw: array<atomic<u32>>;

var<workgroup> wg_near0_count: atomic<u32>;
var<workgroup> wg_near1_count: atomic<u32>;
var<workgroup> wg_far_count: atomic<u32>;
var<workgroup> wg_near0_base: u32;
var<workgroup> wg_near1_base: u32;
var<workgroup> wg_far_base: u32;
var<workgroup> wg_cull: u32;
var<workgroup> wg_keep_lanes: u32;
var<workgroup> wg_in_wake: u32;
var<workgroup> wg_in_shadow: u32;
var<workgroup> wg_light_cnt: u32;
var<workgroup> wg_light_ids: array<u32, 33>;
var<workgroup> wg_cam_r: vec2<f32>;

fn keep_prob_at(dist: f32) -> f32 {
    let near_t = smoothstep(12.0, 28.0, dist);
    let mid_t = smoothstep(28.0, 58.0, dist);
    let far_t = smoothstep(58.0, 105.0, dist);
    let p1 = mix(1.0, 0.52, near_t);
    let p2 = mix(p1, 0.24, mid_t);
    return mix(p2, 0.12, far_t);
}

fn plane_signed_dist(plane: vec4<f32>, c: vec3<f32>) -> f32 {
    let inv_l = inverseSqrt(max(dot(plane.xyz, plane.xyz), 1.0e-12));
    return (dot(plane.xyz, c) + plane.w) * inv_l;
}

fn classify_sphere(c: vec3<f32>, r: f32) -> u32 {
    let m = world.view_proj;
    let r0 = vec4<f32>(m[0].x, m[1].x, m[2].x, m[3].x);
    let r1 = vec4<f32>(m[0].y, m[1].y, m[2].y, m[3].y);
    let r2 = vec4<f32>(m[0].z, m[1].z, m[2].z, m[3].z);
    let r3 = vec4<f32>(m[0].w, m[1].w, m[2].w, m[3].w);

    let d_l = plane_signed_dist(r3 + r0, c);
    let d_r = plane_signed_dist(r3 - r0, c);
    let d_b = plane_signed_dist(r3 + r1, c);
    let d_t = plane_signed_dist(r3 - r1, c);
    let d_f = plane_signed_dist(r2, c);
    let d_n = plane_signed_dist(r3 - r2, c);

    let min_d = min(min(min(d_l, d_r), min(d_b, d_t)), min(d_f, d_n));
    if min_d < -r {
        return 0u;
    }
    if min_d > r {
        return 2u;
    }
    return 1u;
}

fn patch_has_wake(xz: vec2<f32>, rad: f32) -> u32 {
    if abs(xz.x) >= WAKE_EXTENT + rad || abs(xz.y) >= WAKE_EXTENT + rad {
        return 0u;
    }
    let lo_uv = clamp((xz - vec2<f32>(rad)) / (WAKE_EXTENT * 2.0) + 0.5, vec2<f32>(0.0), vec2<f32>(0.999));
    let hi_uv = clamp((xz + vec2<f32>(rad)) / (WAKE_EXTENT * 2.0) + 0.5, vec2<f32>(0.0), vec2<f32>(0.999));
    let t0 = vec2<u32>(floor(lo_uv * 16.0));
    let t1 = min(vec2<u32>(floor(hi_uv * 16.0)), vec2<u32>(15u));
    for (var ty = t0.y; ty <= t1.y; ty = ty + 1u) {
        for (var tx = t0.x; tx <= t1.x; tx = tx + 1u) {
            if wake[65536u + ty * 16u + tx].bend.x > 0.5 {
                return 1u;
            }
        }
    }
    return 0u;
}

fn sample_wake(xz: vec2<f32>) -> vec3<f32> {
    if abs(xz.x) >= WAKE_EXTENT || abs(xz.y) >= WAKE_EXTENT {
        return vec3<f32>(0.0);
    }
    let uv = xz / (WAKE_EXTENT * 2.0) + 0.5;
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

fn grass_dynamic_lights_clustered(pos: vec3<f32>, cnt: u32) -> vec3<f32> {
    var accum = vec3<f32>(0.0);
    for (var k = 0u; k < cnt; k = k + 1u) {
        let i = wg_light_ids[k];
        let pr = lights[i].pos_radius;
        let ci = lights[i].color_intensity;
        let d_vec = pr.xyz - pos;
        let d2 = dot(d_vec, d_vec);
        let r2 = pr.w * pr.w;
        if d2 >= r2 {
            continue;
        }
        let d = sqrt(max(d2, 0.001));
        let u = 1.0 - d / pr.w;
        let atten = (u * u) / (1.0 + 0.45 * max(d2, 0.45));
        accum += ci.rgb * (ci.w * atten);
    }
    return accum;
}

@compute @workgroup_size(256)
fn cs_cull_grass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    if lid == 0u {
        atomicStore(&wg_near0_count, 0u);
        atomicStore(&wg_near1_count, 0u);
        atomicStore(&wg_far_count, 0u);
        var cull_state = 0u;
        var keep_lanes = 256u;
        var in_wake = 0u;
        var in_shadow = 0u;
        var l_cnt = 0u;
        if wid.x < arrayLength(&patch_bounds) {
            let pb = patch_bounds[wid.x].sphere;
            if pb.w > 0.0 {
                let p_dist = length(pb.xyz - world.camera_pos.xyz);
                let min_d = max(p_dist - pb.w, 0.0);
                if min_d <= 112.0 {
                    cull_state = classify_sphere(pb.xyz, pb.w);
                    if cull_state != 0u {
                        let max_kp = keep_prob_at(min_d);
                        keep_lanes = min(u32(ceil(max_kp * 256.0)), 256u);
                        in_wake = patch_has_wake(pb.xz, pb.w);
                        let s_xz = world.sun_dir.xz / max(world.sun_dir.y, 0.26);
                        let sh_hit = pb.xz + s_xz * max(9.8 - pb.y, 0.0);
                        let sh_d = length(sh_hit - vec2<f32>(0.5, -1.8));
                        in_shadow = u32(sh_d < 16.0 + pb.w * 2.2);
                        let has_l = (world.sky_zenith.w + world.sky_horizon.w) >= 0.05;
                        if has_l && min_d < 38.0 {
                            let max_i = select(5u, min(arrayLength(&lights), 33u), world.sky_horizon.w > 0.25 && min_d < 31.5);
                            for (var i = 0u; i < max_i; i = i + 1u) {
                                let ci = lights[i].color_intensity;
                                if ci.w <= 0.01 {
                                    continue;
                                }
                                let pr = lights[i].pos_radius;
                                let d_v = pr.xyz - pb.xyz;
                                let sr = pr.w + pb.w;
                                if dot(d_v, d_v) < sr * sr {
                                    wg_light_ids[l_cnt] = i;
                                    l_cnt += 1u;
                                }
                            }
                        }
                    }
                }
            }
        }
        wg_cull = cull_state;
        wg_keep_lanes = keep_lanes;
        wg_in_wake = in_wake;
        wg_in_shadow = in_shadow;
        wg_light_cnt = l_cnt;
        wg_cam_r = normalize(vec2<f32>(world.camera_right.x, world.camera_right.z) + vec2<f32>(0.0001, 0.0));
    }
    workgroupBarrier();

    let idx = gid.x;
    let active_count = min(u32(world.time_wind.w + 0.5), arrayLength(&blades));
    var keep_lod = 0u;
    var local_slot = 0u;
    var vis_out = VisibleBlade(0.0, 0.0, 0.0, 0u, vec4<u32>(0u));

    if wg_cull != 0u && lid < wg_keep_lanes && idx < active_count {
        let blade = blades[idx];
        let packed = blade.packed;
        if packed != 0u {
            let root = vec3<f32>(blade.x, blade.y, blade.z);
            let delta = root - world.camera_pos.xyz;
            let dist_sq = dot(delta, delta);
            if dist_sq <= 12500.0 {
                let height = f32(packed & 511u) * (3.6 / 511.0);
                var in_frustum = wg_cull == 2u;
                if !in_frustum {
                    let root_clip = world.view_proj * vec4<f32>(root.x, root.y + height * 0.5, root.z, 1.0);
                    let margin = root_clip.w * 1.12 + 1.6;
                    in_frustum = root_clip.w >= -1.2 && abs(root_clip.x) <= margin && abs(root_clip.y) <= margin;
                }
                if in_frustum {
                    let dist = sqrt(max(dist_sq, 0.001));
                    let keep_prob = keep_prob_at(dist);
                    let thin_u = (f32(lid) + 0.5) * (1.0 / 256.0);
                    if thin_u <= keep_prob {
                        let yaw = f32((packed >> 9u) & 127u) * (TAU / 127.0);
                        let patch_n = f32((packed >> 16u) & 63u) * (1.0 / 63.0);
                        let seed_val = f32((packed >> 22u) & 127u) * (1.0 / 127.0);
                        let flower_q = (packed >> 29u) & 7u;

                        let time = world.time_wind.x;
                        let wind_str = world.time_wind.z;
                        let w_vec = sample_wind(root.xz, time, wind_str);
                        var flutter = 0.0;
                        if dist < 28.0 {
                            flutter = sin(time * (4.2 + wind_str * 1.6) + seed_val * 25.0 + root.x * 1.3) * 0.08 * (0.4 + wind_str);
                        }
                        var wake_v = vec3<f32>(0.0);
                        if wg_in_wake != 0u {
                            wake_v = sample_wake(root.xz);
                        }

                        let blade_dir = vec2<f32>(cos(yaw), sin(yaw));
                        let arch_lean = blade_dir * (0.22 + 0.24 * seed_val);
                        let bend_xz = arch_lean + w_vec * (0.42 + 0.20 * seed_val) + vec2<f32>(flutter, flutter * 0.6) + wake_v.xy * 1.35;
                        let bend_len = length(bend_xz);
                        let droop = min(bend_len * 0.44 + wake_v.z * 0.72, 0.82);

                        let blade_right_xz = normalize(mix(vec2<f32>(-blade_dir.y, blade_dir.x), wg_cam_r, smoothstep(7.0, 26.0, dist) * 0.72));

                        let dist_w = 1.0 + smoothstep(14.0, 66.0, dist) * 1.45;
                        let comp_w = mix(1.0, inverseSqrt(max(keep_prob, 0.12)), 0.52);
                        let base_width = (0.034 + 0.016 * seed_val) * dist_w * comp_w;

                        let mid_pos = vec3<f32>(
                            root.x + bend_xz.x * height * 0.35,
                            root.y + height * 0.55,
                            root.z + bend_xz.y * height * 0.35,
                        );
                        var shadow = 1.0;
                        if wg_in_shadow != 0u {
                            shadow = canopy_shadow(mid_pos, world.sun_dir.xyz);
                        }
                        var dyn_light = vec3<f32>(0.0);
                        if wg_light_cnt > 0u && dist_sq < 1400.0 {
                            dyn_light = grass_dynamic_lights_clustered(mid_pos, wg_light_cnt);
                        }
                        let wind_mag = clamp(length(w_vec), 0.0, 2.4);

                        let dyn_enc = sqrt(clamp(dyn_light * 0.0625, vec3<f32>(0.0), vec3<f32>(1.0)));
                        let p_q = u32(clamp(patch_n, 0.0, 1.0) * 127.0 + 0.5) & 127u;
                        let s_q = u32(clamp(seed_val, 0.0, 1.0) * 127.0 + 0.5) & 127u;
                        let sh_q = u32(clamp(shadow, 0.0, 1.0) * 127.0 + 0.5) & 127u;
                        let w_q = u32(clamp(wind_mag * (1.0 / 2.4), 0.0, 1.0) * 255.0 + 0.5) & 255u;
                        let meta_u = p_q | (s_q << 7u) | (sh_q << 14u) | (w_q << 21u) | ((flower_q & 7u) << 29u);

                        vis_out = VisibleBlade(
                            root.x,
                            root.y,
                            root.z,
                            pack2x16float(vec2<f32>(height, base_width)),
                            vec4<u32>(
                                pack2x16float(bend_xz),
                                pack2x16snorm(blade_right_xz),
                                pack4x8unorm(vec4<f32>(dyn_enc, droop)),
                                meta_u,
                            ),
                        );

                        let lod_dist = select(16.0 + seed_val * 8.0, 28.0, flower_q > 0u);
                        if dist < 13.0 {
                            keep_lod = 1u;
                            local_slot = atomicAdd(&wg_near0_count, 1u);
                        } else if dist < lod_dist {
                            keep_lod = 2u;
                            local_slot = atomicAdd(&wg_near1_count, 1u);
                        } else {
                            keep_lod = 3u;
                            local_slot = atomicAdd(&wg_far_count, 1u);
                        }
                    }
                }
            }
        }
    }

    workgroupBarrier();
    if lid == 0u && wg_cull != 0u {
        let n0 = atomicLoad(&wg_near0_count);
        if n0 > 0u {
            wg_near0_base = atomicAdd(&grass_args_rw[1], n0);
        }
        let n1 = atomicLoad(&wg_near1_count);
        if n1 > 0u {
            wg_near1_base = atomicAdd(&grass_args_rw[5], n1);
        }
        let f_cnt = atomicLoad(&wg_far_count);
        if f_cnt > 0u {
            wg_far_base = atomicAdd(&grass_args_rw[9], f_cnt);
        }
    }
    workgroupBarrier();

    if keep_lod == 1u {
        let dst = wg_near0_base + local_slot;
        if dst < arrayLength(&vis_near0_rw) {
            vis_near0_rw[dst] = vis_out;
        }
    } else if keep_lod == 2u {
        let dst = wg_near1_base + local_slot;
        if dst < arrayLength(&vis_near1_rw) {
            vis_near1_rw[dst] = vis_out;
        }
    } else if keep_lod == 3u {
        let dst = wg_far_base + local_slot;
        if dst < arrayLength(&vis_far_rw) {
            vis_far_rw[dst] = vis_out;
        }
    }
}
