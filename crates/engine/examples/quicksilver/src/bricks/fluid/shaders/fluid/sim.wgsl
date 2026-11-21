import package::fluid::common::{
    BRICK_X, BRICK_Y, BRICK_Z, NUM_BRICKS, SimParams, brick_idx, cell_idx, decode_fixed,
    encode_fixed, pack_grid_vel, pack_meta, rand01, unpack_grid_vel, unpack_meta,
};

@group(1) @binding(0) var<uniform> sim_params: SimParams;
@group(1) @binding(1) var<storage, read_write> sim_particles: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> sim_particles_tmp: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> grid_atomic: array<atomic<i32>>;
@group(1) @binding(4) var<storage, read_write> grid_vel: array<vec2<u32>>;
@group(1) @binding(5) var<storage, read_write> heightfield_out: array<vec4<f32>>;
@group(1) @binding(6) var<storage, read_write> draw_args: array<atomic<u32>>;
@group(1) @binding(7) var<storage, read_write> surface_indices: array<u32>;
@group(1) @binding(8) var<storage, read_write> brick_counts: array<atomic<u32>>;
@group(1) @binding(9) var<storage, read_write> chunk_list: array<vec4<u32>>;
@group(1) @binding(10) var<storage, read_write> dispatch_args: array<u32>;

var<workgroup> wg_surf_count: atomic<u32>;
var<workgroup> wg_surf_base: u32;
var<workgroup> wg_scan_p_a: array<u32, 256>;
var<workgroup> wg_scan_p_b: array<u32, 256>;
var<workgroup> wg_scan_c_a: array<u32, 256>;
var<workgroup> wg_scan_c_b: array<u32, 256>;
var<workgroup> sh_vel_6: array<vec2<u32>, 216>;
var<workgroup> sh_vel_8: array<vec2<u32>, 512>;
var<workgroup> sh_atom: array<atomic<i32>, 2048>;
var<workgroup> sh_cell_seen: array<atomic<u32>, 128>;

fn bspline_weights(fx: vec3<f32>) -> array<vec3<f32>, 3> {
    let w0 = 0.5 * (1.5 - fx) * (1.5 - fx);
    let w1 = 0.75 - (fx - 1.0) * (fx - 1.0);
    let w2 = 0.5 * (fx - 0.5) * (fx - 0.5);
    return array<vec3<f32>, 3>(w0, w1, w2);
}

fn cell_brick(pos: vec3<f32>) -> u32 {
    let dims = sim_params.dims.xyz;
    let cell = clamp(vec3<i32>(floor(pos - 0.5)), vec3<i32>(0), vec3<i32>(dims) - vec3<i32>(3));
    let bx = min(u32(cell.x) >> 2u, BRICK_X - 1u);
    let by = min(u32(cell.y) >> 2u, BRICK_Y - 1u);
    let bz = min(u32(cell.z) >> 2u, BRICK_Z - 1u);
    return brick_idx(bx, by, bz);
}

fn p2g_affine(c: mat3x3<f32>, density: f32) -> mat3x3<f32> {
    let dt = sim_params.physics.x;
    let mass = sim_params.physics.y;
    let rest_density = sim_params.physics.z;
    let stiffness = sim_params.physics.w;
    let viscosity = sim_params.fluid.x;

    let d_safe = max(density, 0.35);
    let ratio = clamp(d_safe / rest_density, 0.2, 2.6);
    let eq_p = stiffness * (ratio * ratio * ratio - 1.0);
    let pressure = clamp(eq_p, -0.06, 85.0);
    let volume = mass / d_safe;

    let strain = c + transpose(c);
    let stress = mat3x3<f32>(
        vec3<f32>(-pressure, 0.0, 0.0),
        vec3<f32>(0.0, -pressure, 0.0),
        vec3<f32>(0.0, 0.0, -pressure),
    ) + viscosity * strain;
    return mass * c - (volume * 4.0 * dt) * stress;
}

fn p2g_scatter_global(pos: vec3<f32>, vel: vec3<f32>, c: mat3x3<f32>, density: f32) {
    let mass = sim_params.physics.y;
    let dims = sim_params.dims.xyz;
    let affine = p2g_affine(c, density);
    let mv = mass * vel;

    let cell = clamp(vec3<i32>(floor(pos - 0.5)), vec3<i32>(0), vec3<i32>(dims) - vec3<i32>(3));
    let ucell = vec3<u32>(cell);
    let fx = pos - vec3<f32>(cell);
    let w = bspline_weights(fx);

    for (var oz = 0u; oz < 3u; oz += 1u) {
        let wz = w[oz].z;
        let dz = f32(oz) - fx.z;
        for (var oy = 0u; oy < 3u; oy += 1u) {
            let wyz = w[oy].y * wz;
            let dy = f32(oy) - fx.y;
            let base_mom = mv + affine[1] * dy + affine[2] * dz;
            for (var ox = 0u; ox < 3u; ox += 1u) {
                let weight = w[ox].x * wyz;
                if weight < 0.0022 {
                    continue;
                }
                let dx = f32(ox) - fx.x;
                let mw = mass * weight;
                let mom = (base_mom + affine[0] * dx) * weight;
                let base = cell_idx(ucell.x + ox, ucell.y + oy, ucell.z + oz) << 2u;
                let mx = encode_fixed(mom.x);
                let my = encode_fixed(mom.y);
                let mz = encode_fixed(mom.z);
                let mm = encode_fixed(mw);
                if mx != 0 { atomicAdd(&grid_atomic[base], mx); }
                if my != 0 { atomicAdd(&grid_atomic[base + 1u], my); }
                if mz != 0 { atomicAdd(&grid_atomic[base + 2u], mz); }
                atomicAdd(&grid_atomic[base + 3u], mm);
            }
        }
    }
}

fn p2g_scatter_shared(
    pos: vec3<f32>,
    vel: vec3<f32>,
    c: mat3x3<f32>,
    density: f32,
    origin: vec3<i32>,
    lid: u32,
) {
    let dims = sim_params.dims.xyz;
    let cell = clamp(vec3<i32>(floor(pos - 0.5)), vec3<i32>(0), vec3<i32>(dims) - vec3<i32>(3));
    let rel = cell - origin;
    if rel.x < 0 || rel.y < 0 || rel.z < 0 || rel.x > 5 || rel.y > 5 || rel.z > 5 {
        p2g_scatter_global(pos, vel, c, density);
        return;
    }

    let mass = sim_params.physics.y;
    let affine = p2g_affine(c, density);
    let mv = mass * vel;
    let fx = pos - vec3<f32>(cell);
    let w = bspline_weights(fx);
    let ur = vec3<u32>(rel);

    let sx = lid % 3u;
    let sy = (lid / 3u) % 3u;
    let sz = (lid / 9u) % 3u;

    for (var iz = 0u; iz < 3u; iz += 1u) {
        let tz = iz + sz;
        let oz = select(tz, tz - 3u, tz >= 3u);
        let wz = w[oz].z;
        let dz = f32(oz) - fx.z;
        let lz = (ur.z + oz) << 6u;
        for (var iy = 0u; iy < 3u; iy += 1u) {
            let ty = iy + sy;
            let oy = select(ty, ty - 3u, ty >= 3u);
            let wyz = w[oy].y * wz;
            let dy = f32(oy) - fx.y;
            let lzy = lz | ((ur.y + oy) << 3u);
            let base_mom = mv + affine[1] * dy + affine[2] * dz;
            for (var ix = 0u; ix < 3u; ix += 1u) {
                let tx = ix + sx;
                let ox = select(tx, tx - 3u, tx >= 3u);
                let weight = w[ox].x * wyz;
                if weight < 0.0022 {
                    continue;
                }
                let dx = f32(ox) - fx.x;
                let mw = mass * weight;
                let mom = (base_mom + affine[0] * dx) * weight;
                let base = (lzy | (ur.x + ox)) << 2u;
                let mx = encode_fixed(mom.x);
                let my = encode_fixed(mom.y);
                let mz = encode_fixed(mom.z);
                let mm = encode_fixed(mw);
                if mx != 0 { atomicAdd(&sh_atom[base], mx); }
                if my != 0 { atomicAdd(&sh_atom[base + 1u], my); }
                if mz != 0 { atomicAdd(&sh_atom[base + 2u], mz); }
                atomicAdd(&sh_atom[base + 3u], mm);
            }
        }
    }
}

fn flush_shared_atom(origin: vec3<i32>, lid: u32) {
    let dims = vec3<i32>(sim_params.dims.xyz);
    for (var cidx = lid; cidx < 512u; cidx += 128u) {
        let sb = cidx << 2u;
        let ax = atomicLoad(&sh_atom[sb]);
        let ay = atomicLoad(&sh_atom[sb + 1u]);
        let az = atomicLoad(&sh_atom[sb + 2u]);
        let am = atomicLoad(&sh_atom[sb + 3u]);
        if (ax | ay | az | am) == 0 {
            continue;
        }
        let gx = origin.x + i32(cidx & 7u);
        let gy = origin.y + i32((cidx >> 3u) & 7u);
        let gz = origin.z + i32(cidx >> 6u);
        if gx >= 0 && gx < dims.x && gy >= 0 && gy < dims.y && gz >= 0 && gz < dims.z {
            let gb = cell_idx(u32(gx), u32(gy), u32(gz)) << 2u;
            if ax != 0 { atomicAdd(&grid_atomic[gb], ax); }
            if ay != 0 { atomicAdd(&grid_atomic[gb + 1u], ay); }
            if az != 0 { atomicAdd(&grid_atomic[gb + 2u], az); }
            if am != 0 { atomicAdd(&grid_atomic[gb + 3u], am); }
        }
    }
}

fn brick_origin(brick: u32) -> vec3<i32> {
    let bx = brick % BRICK_X;
    let by = (brick / BRICK_X) % BRICK_Y;
    let bz = brick / (BRICK_X * BRICK_Y);
    return vec3<i32>(i32(bx << 2u), i32(by << 2u), i32(bz << 2u));
}

@compute @workgroup_size(128)
fn init_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let count = sim_params.dims.w;
    if i >= count {
        return;
    }
    let gx = f32(sim_params.dims.x);
    let gz = f32(sim_params.dims.z);
    let rest_d = sim_params.physics.z;
    let r0 = rand01(i * 1973u + 17u);
    let r1 = rand01(i * 9277u + 31u);
    let r2 = rand01(i * 26699u + 53u);
    let share = f32(i) / f32(count);

    var pos = vec3<f32>(0.0);
    var vel = vec3<f32>(0.0);

    if share < 0.64 {
        pos = vec3<f32>(
            3.3 + r0 * (gx - 6.6),
            3.2 + r1 * 7.1,
            3.3 + r2 * (gz - 6.6),
        );
        vel = vec3<f32>(0.12, 0.0, 0.0);
    } else if share < 0.96 {
        let crest = sin(r2 * 3.1415926) * 2.2;
        pos = vec3<f32>(
            3.3 + r0 * (gx * 0.32),
            10.3 + r1 * (9.2 + crest),
            3.3 + r2 * (gz - 6.6),
        );
        let top_boost = 0.65 + 0.75 * r1;
        vel = vec3<f32>(1.28 * top_boost, -0.06, (r2 - 0.5) * 0.14);
    } else {
        let u = r0 * 6.2831853;
        let cos_t = r1 * 2.0 - 1.0;
        let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
        let rad = 6.8 * pow(r2, 0.3333333);
        let dir = vec3<f32>(sin_t * cos(u), cos_t, sin_t * sin(u));
        pos = vec3<f32>(gx * 0.66, 17.6, gz * 0.50) + dir * rad;
        vel = vec3<f32>(-0.32, -2.05, 0.0);
    }

    sim_particles[i] = vec4<f32>(pos, pack_meta(0.0, 0.0, 1.0, length(vel)));
    let zero_c = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    p2g_scatter_global(pos, vel, zero_c, rest_d);
}

@compute @workgroup_size(256)
fn clear_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total = sim_params.dims.x * sim_params.dims.y * sim_params.dims.z;
    if idx >= total {
        return;
    }
    let base = idx << 2u;
    atomicStore(&grid_atomic[base], 0);
    atomicStore(&grid_atomic[base + 1u], 0);
    atomicStore(&grid_atomic[base + 2u], 0);
    atomicStore(&grid_atomic[base + 3u], 0);
    grid_vel[idx] = vec2<u32>(0u);
}

@compute @workgroup_size(256)
fn bin_clear(@builtin(global_invocation_id) id: vec3<u32>) {
    let b = id.x;
    if b < NUM_BRICKS {
        atomicStore(&brick_counts[b], 0u);
    }
}

@compute @workgroup_size(128)
fn bin_count(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= sim_params.dims.w {
        return;
    }
    let b = cell_brick(sim_particles[i].xyz);
    atomicAdd(&brick_counts[b], 1u);
}

@compute @workgroup_size(256)
fn bin_prefix_chunks(@builtin(local_invocation_index) tid: u32) {
    let base = tid * 14u;
    var local_p = 0u;
    var local_c = 0u;
    for (var k = 0u; k < 14u; k += 1u) {
        let b = base + k;
        if b < NUM_BRICKS {
            let cnt = atomicLoad(&brick_counts[b]);
            local_p += cnt;
            local_c += (cnt + 127u) >> 7u;
        }
    }
    wg_scan_p_a[tid] = local_p;
    wg_scan_c_a[tid] = local_c;
    workgroupBarrier();

    let i1 = select(0u, tid - 1u, tid >= 1u);
    wg_scan_p_b[tid] = wg_scan_p_a[tid] + select(0u, wg_scan_p_a[i1], tid >= 1u);
    wg_scan_c_b[tid] = wg_scan_c_a[tid] + select(0u, wg_scan_c_a[i1], tid >= 1u);
    workgroupBarrier();
    let i2 = select(0u, tid - 2u, tid >= 2u);
    wg_scan_p_a[tid] = wg_scan_p_b[tid] + select(0u, wg_scan_p_b[i2], tid >= 2u);
    wg_scan_c_a[tid] = wg_scan_c_b[tid] + select(0u, wg_scan_c_b[i2], tid >= 2u);
    workgroupBarrier();
    let i4 = select(0u, tid - 4u, tid >= 4u);
    wg_scan_p_b[tid] = wg_scan_p_a[tid] + select(0u, wg_scan_p_a[i4], tid >= 4u);
    wg_scan_c_b[tid] = wg_scan_c_a[tid] + select(0u, wg_scan_c_a[i4], tid >= 4u);
    workgroupBarrier();
    let i8 = select(0u, tid - 8u, tid >= 8u);
    wg_scan_p_a[tid] = wg_scan_p_b[tid] + select(0u, wg_scan_p_b[i8], tid >= 8u);
    wg_scan_c_a[tid] = wg_scan_c_b[tid] + select(0u, wg_scan_c_b[i8], tid >= 8u);
    workgroupBarrier();
    let i16 = select(0u, tid - 16u, tid >= 16u);
    wg_scan_p_b[tid] = wg_scan_p_a[tid] + select(0u, wg_scan_p_a[i16], tid >= 16u);
    wg_scan_c_b[tid] = wg_scan_c_a[tid] + select(0u, wg_scan_c_a[i16], tid >= 16u);
    workgroupBarrier();
    let i32_i = select(0u, tid - 32u, tid >= 32u);
    wg_scan_p_a[tid] = wg_scan_p_b[tid] + select(0u, wg_scan_p_b[i32_i], tid >= 32u);
    wg_scan_c_a[tid] = wg_scan_c_b[tid] + select(0u, wg_scan_c_b[i32_i], tid >= 32u);
    workgroupBarrier();
    let i64 = select(0u, tid - 64u, tid >= 64u);
    wg_scan_p_b[tid] = wg_scan_p_a[tid] + select(0u, wg_scan_p_a[i64], tid >= 64u);
    wg_scan_c_b[tid] = wg_scan_c_a[tid] + select(0u, wg_scan_c_a[i64], tid >= 64u);
    workgroupBarrier();
    let i128 = select(0u, tid - 128u, tid >= 128u);
    wg_scan_p_a[tid] = wg_scan_p_b[tid] + select(0u, wg_scan_p_b[i128], tid >= 128u);
    wg_scan_c_a[tid] = wg_scan_c_b[tid] + select(0u, wg_scan_c_b[i128], tid >= 128u);
    workgroupBarrier();

    if tid == 255u {
        dispatch_args[0] = min(wg_scan_c_a[255], 8192u);
        dispatch_args[1] = 1u;
        dispatch_args[2] = 1u;
    }

    var p_run = select(0u, wg_scan_p_a[i1], tid > 0u);
    var c_run = select(0u, wg_scan_c_a[i1], tid > 0u);
    for (var k = 0u; k < 14u; k += 1u) {
        let b = base + k;
        if b < NUM_BRICKS {
            let cnt = atomicLoad(&brick_counts[b]);
            atomicStore(&brick_counts[b], p_run);
            var rem = cnt;
            var off = p_run;
            while rem > 0u {
                let take = min(rem, 128u);
                if c_run < 8192u {
                    chunk_list[c_run] = vec4<u32>(b, off, take, 0u);
                }
                c_run += 1u;
                off += take;
                rem -= take;
            }
            p_run += cnt;
        }
    }
}

@compute @workgroup_size(128)
fn bin_scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= sim_params.dims.w {
        return;
    }
    let p = sim_particles[i];
    let b = cell_brick(p.xyz);
    let dst = atomicAdd(&brick_counts[b], 1u);
    sim_particles_tmp[dst] = p;
}

@compute @workgroup_size(256)
fn update_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx == 0u {
        atomicStore(&draw_args[0], 6u);
        atomicStore(&draw_args[1], 0u);
        atomicStore(&draw_args[2], 0u);
        atomicStore(&draw_args[3], 0u);
    }
    let dims = sim_params.dims.xyz;
    let total = dims.x * dims.y * dims.z;
    if idx >= total {
        return;
    }
    let base = idx << 2u;
    let ax = atomicLoad(&grid_atomic[base]);
    let ay = atomicLoad(&grid_atomic[base + 1u]);
    let az = atomicLoad(&grid_atomic[base + 2u]);
    let am = atomicLoad(&grid_atomic[base + 3u]);
    if (ax | ay | az | am) == 0 {
        let prev = grid_vel[idx];
        if (prev.x | prev.y) != 0u {
            grid_vel[idx] = vec2<u32>(0u);
        }
        return;
    }
    atomicStore(&grid_atomic[base], 0);
    atomicStore(&grid_atomic[base + 1u], 0);
    atomicStore(&grid_atomic[base + 2u], 0);
    atomicStore(&grid_atomic[base + 3u], 0);

    let mass = decode_fixed(am);
    if mass <= 1.0e-4 {
        grid_vel[idx] = vec2<u32>(0u);
        return;
    }

    let local = idx & 63u;
    let brick = idx >> 6u;
    let bx = brick % BRICK_X;
    let by = (brick / BRICK_X) % BRICK_Y;
    let bz = brick / (BRICK_X * BRICK_Y);
    let gx = (bx << 2u) | (local & 3u);
    let gy = (by << 2u) | ((local >> 2u) & 3u);
    let gz = (bz << 2u) | (local >> 4u);
    let pos = vec3<f32>(f32(gx), f32(gy), f32(gz));
    let dt = sim_params.physics.x;
    let gravity = sim_params.fluid.y;
    let vortex = sim_params.fluid.z;
    let zero_g = sim_params.fluid.w;
    let elapsed = sim_params.time_info.x;

    var vel = vec3<f32>(decode_fixed(ax), decode_fixed(ay), decode_fixed(az)) / mass;
    let center = vec3<f32>(f32(dims.x) * 0.5, f32(dims.y) * 0.56, f32(dims.z) * 0.5);

    if zero_g > 0.5 {
        let wobble = vec3<f32>(
            sin(elapsed * 1.15) * 1.3,
            cos(elapsed * 0.92) * 1.1,
            sin(elapsed * 0.78 + 1.1) * 1.0,
        );
        let to_c = (center + wobble) - pos;
        let dist = length(to_c);
        let pull = to_c * (0.165 / (1.0 + dist * 0.022));
        let tangent = normalize(vec3<f32>(-to_c.z, 0.30 * sin(elapsed * 1.4 + pos.y * 0.14), to_c.x) + vec3<f32>(1.0e-4));
        let swirl = tangent * (0.15 * smoothstep(2.0, 14.0, dist));
        vel += (pull + swirl) * dt;
        vel *= 0.990;
    } else {
        vel.y += gravity * dt;
    }

    if vortex > 0.5 {
        let rel = pos.xz - center.xz;
        let r = length(rel);
        if r > 0.4 {
            let tang = vec2<f32>(-rel.y, rel.x) / r;
            let rad_dir = rel / r;
            let band = smoothstep(0.8, 10.0, r) * (1.0 - smoothstep(24.0, 40.0, r));
            let force_xz = tang * (1.65 * band) - rad_dir * (0.36 * smoothstep(5.0, 26.0, r));
            vel.x += force_xz.x * dt;
            vel.z += force_xz.y * dt;
            if zero_g < 0.5 {
                let rim_lift = smoothstep(12.0, 26.0, r) * (1.0 - smoothstep(14.0, 28.0, pos.y));
                vel.y += rim_lift * 0.24 * dt;
            }
        }
    }

    if sim_params.stir_pos.w > 0.01 {
        let d = pos - sim_params.stir_pos.xyz;
        let rad = max(sim_params.stir_vel.w, 4.0);
        let dist = length(d);
        if dist < rad * 1.5 {
            let falloff = max(1.0 - dist / (rad * 1.5), 0.0);
            let k = falloff * falloff * sim_params.stir_pos.w;
            let push_dir = d / max(dist, 0.25);
            let stir_v = sim_params.stir_vel.xyz;
            let speed = length(stir_v);
            vel += (stir_v * 3.1 + push_dir * (0.50 + speed * 0.60) + vec3<f32>(0.0, 0.30 + speed * 0.42, 0.0)) * k * dt * 3.6;
        }
    }

    let pad = 3u;
    if gx < pad && vel.x < 0.0 { vel.x = 0.0; }
    if gx >= dims.x - pad && vel.x > 0.0 { vel.x = 0.0; }
    if gy < pad && vel.y < 0.0 { vel.y = 0.0; vel.x *= 0.992; vel.z *= 0.992; }
    if gy >= dims.y - pad && vel.y > 0.0 { vel.y = 0.0; }
    if gz < pad && vel.z < 0.0 { vel.z = 0.0; }
    if gz >= dims.z - pad && vel.z > 0.0 { vel.z = 0.0; }

    let speed = length(vel);
    if speed > 2.45 {
        vel = vel * (2.45 / speed);
    }

    grid_vel[idx] = pack_grid_vel(vel, mass);
}

@compute @workgroup_size(128)
fn g2p2g_brick_mid(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let chunk = chunk_list[wg_id.x];
    let b_orig = brick_origin(chunk.x);
    let a_orig = b_orig - vec3<i32>(1);
    let dims = vec3<i32>(sim_params.dims.xyz);

    for (var idx = lid; idx < 216u; idx += 128u) {
        let lx = idx % 6u;
        let ly = (idx / 6u) % 6u;
        let lz = idx / 36u;
        let gx = b_orig.x + i32(lx);
        let gy = b_orig.y + i32(ly);
        let gz = b_orig.z + i32(lz);
        if gx >= 0 && gx < dims.x && gy >= 0 && gy < dims.y && gz >= 0 && gz < dims.z {
            sh_vel_6[idx] = grid_vel[cell_idx(u32(gx), u32(gy), u32(gz))];
        } else {
            sh_vel_6[idx] = vec2<u32>(0u);
        }
    }
    for (var k = lid; k < 2048u; k += 128u) {
        atomicStore(&sh_atom[k], 0);
    }
    workgroupBarrier();

    if lid < chunk.z {
        let pid = chunk.y + lid;
        let p = sim_particles_tmp[pid];
        let dt = sim_params.physics.x;
        let rest_density = sim_params.physics.z;
        let cell = clamp(vec3<i32>(floor(p.xyz - 0.5)), vec3<i32>(0), dims - vec3<i32>(3));
        let rel = clamp(cell - b_orig, vec3<i32>(0), vec3<i32>(3));
        let ur = vec3<u32>(rel);
        let fx = p.xyz - vec3<f32>(cell);
        let w = bspline_weights(fx);

        var vel = vec3<f32>(0.0);
        var b0 = vec3<f32>(0.0);
        var b1 = vec3<f32>(0.0);
        var b2 = vec3<f32>(0.0);
        var grid_dens = 0.0;

        for (var oz = 0u; oz < 3u; oz += 1u) {
            let wz = w[oz].z;
            let dz = f32(oz) - fx.z;
            let z_off = (ur.z + oz) * 36u;
            for (var oy = 0u; oy < 3u; oy += 1u) {
                let wyz = w[oy].y * wz;
                let dy = f32(oy) - fx.y;
                let zy_off = z_off + (ur.y + oy) * 6u + ur.x;
                for (var ox = 0u; ox < 3u; ox += 1u) {
                    let weight = w[ox].x * wyz;
                    let gcell = unpack_grid_vel(sh_vel_6[zy_off + ox]);
                    let dx = f32(ox) - fx.x;
                    let wv = gcell.xyz * weight;
                    vel += wv;
                    b0 += wv * dx;
                    b1 += wv * dy;
                    b2 += wv * dz;
                    grid_dens += gcell.w * weight;
                }
            }
        }

        let c = mat3x3<f32>(b0 * 4.0, b1 * 4.0, b2 * 4.0);
        let div_v = clamp(c[0].x + c[1].y + c[2].z, -3.2, 3.2);
        let pred_dens = clamp(grid_dens / clamp(1.0 + dt * div_v, 0.72, 1.38), 0.3, rest_density * 2.8);

        var next_pos = p.xyz + vel * dt;
        let min_p = vec3<f32>(3.05, 3.05, 3.05);
        let max_p = vec3<f32>(f32(dims.x) - 3.05, f32(dims.y) - 3.05, f32(dims.z) - 3.05);
        if next_pos.x < min_p.x { next_pos.x = min_p.x; vel.x = max(vel.x, 0.0); }
        if next_pos.x > max_p.x { next_pos.x = max_p.x; vel.x = min(vel.x, 0.0); }
        if next_pos.y < min_p.y { next_pos.y = min_p.y; vel.y = max(vel.y, 0.0); }
        if next_pos.y > max_p.y { next_pos.y = max_p.y; vel.y = min(vel.y, 0.0); }
        if next_pos.z < min_p.z { next_pos.z = min_p.z; vel.z = max(vel.z, 0.0); }
        if next_pos.z > max_p.z { next_pos.z = max_p.z; vel.z = min(vel.z, 0.0); }

        sim_particles[pid] = vec4<f32>(next_pos, p.w);
        p2g_scatter_shared(next_pos, vel, c, pred_dens, a_orig, lid);
    }

    workgroupBarrier();
    flush_shared_atom(a_orig, lid);
}

@compute @workgroup_size(128)
fn g2p2g_brick_final(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let chunk = chunk_list[wg_id.x];
    let a_orig = brick_origin(chunk.x) - vec3<i32>(1);
    let dims = vec3<i32>(sim_params.dims.xyz);

    for (var idx = lid; idx < 512u; idx += 128u) {
        let gx = a_orig.x + i32(idx & 7u);
        let gy = a_orig.y + i32((idx >> 3u) & 7u);
        let gz = a_orig.z + i32(idx >> 6u);
        if gx >= 0 && gx < dims.x && gy >= 0 && gy < dims.y && gz >= 0 && gz < dims.z {
            sh_vel_8[idx] = grid_vel[cell_idx(u32(gx), u32(gy), u32(gz))];
        } else {
            sh_vel_8[idx] = vec2<u32>(0u);
        }
    }
    atomicStore(&sh_cell_seen[lid], 0u);
    for (var k = lid; k < 2048u; k += 128u) {
        atomicStore(&sh_atom[k], 0);
    }
    if lid == 0u {
        atomicStore(&wg_surf_count, 0u);
    }
    workgroupBarrier();

    var emit_surface = false;
    var out_pid = 0u;

    if lid < chunk.z {
        let pid = chunk.y + lid;
        out_pid = pid;
        let p = sim_particles[pid];
        let prev_meta = unpack_meta(p.w);
        let dt = sim_params.physics.x;
        let rest_density = sim_params.physics.z;
        let is_drop = sim_params.drop_info.y > 0u && ((pid + sim_params.drop_info.x) % 22u) == 0u;

        if is_drop {
            let k = pid / 22u;
            let seed = sim_params.drop_info.z;
            let gx = f32(dims.x);
            let gy = f32(dims.y);
            let gz = f32(dims.z);
            let cx = gx * (0.36 + 0.28 * rand01(seed + 3u));
            let cz = gz * (0.36 + 0.28 * rand01(seed + 11u));
            let r0 = rand01(k * 3571u + seed);
            let r1 = rand01(k * 7919u + seed + 19u);
            let r2 = rand01(k * 14591u + seed + 47u);
            let u = r0 * 6.2831853;
            let cos_t = r1 * 2.0 - 1.0;
            let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
            let rad = 7.4 * pow(r2, 0.3333333);
            let dir = vec3<f32>(sin_t * cos(u), cos_t, sin_t * sin(u));
            let next_pos = clamp(
                vec3<f32>(cx, 18.5, cz) + dir * rad,
                vec3<f32>(4.0),
                vec3<f32>(gx - 4.5, gy - 4.0, gz - 4.5),
            );
            let vel = vec3<f32>((rand01(seed + 29u) - 0.5) * 0.30, -2.25, (rand01(seed + 37u) - 0.5) * 0.22);
            let zero_c = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
            emit_surface = rad > 5.4;
            sim_particles[pid] = vec4<f32>(next_pos, pack_meta(0.0, 0.35, 1.0, length(vel)));
            p2g_scatter_global(next_pos, vel, zero_c, rest_density);
        } else {
            let cell = clamp(vec3<i32>(floor(p.xyz - 0.5)), vec3<i32>(0), dims - vec3<i32>(3));
            let rel = clamp(cell - a_orig, vec3<i32>(0), vec3<i32>(5));
            let ur = vec3<u32>(rel);
            let fx = p.xyz - vec3<f32>(cell);
            let w = bspline_weights(fx);

            var vel = vec3<f32>(0.0);
            var b0 = vec3<f32>(0.0);
            var b1 = vec3<f32>(0.0);
            var b2 = vec3<f32>(0.0);
            var grid_dens = 0.0;

            for (var oz = 0u; oz < 3u; oz += 1u) {
                let wz = w[oz].z;
                let dz = f32(oz) - fx.z;
                let lz = (ur.z + oz) << 6u;
                for (var oy = 0u; oy < 3u; oy += 1u) {
                    let wyz = w[oy].y * wz;
                    let dy = f32(oy) - fx.y;
                    let lzy = lz | ((ur.y + oy) << 3u);
                    for (var ox = 0u; ox < 3u; ox += 1u) {
                        let weight = w[ox].x * wyz;
                        let gcell = unpack_grid_vel(sh_vel_8[lzy | (ur.x + ox)]);
                        let dx = f32(ox) - fx.x;
                        let wv = gcell.xyz * weight;
                        vel += wv;
                        b0 += wv * dx;
                        b1 += wv * dy;
                        b2 += wv * dz;
                        grid_dens += gcell.w * weight;
                    }
                }
            }

            let c = mat3x3<f32>(b0 * 4.0, b1 * 4.0, b2 * 4.0);
            let div_v = clamp(c[0].x + c[1].y + c[2].z, -3.2, 3.2);
            let pred_dens = clamp(grid_dens / clamp(1.0 + dt * div_v, 0.72, 1.38), 0.3, rest_density * 2.8);

            var next_pos = p.xyz + vel * dt;
            let min_p = vec3<f32>(3.05, 3.05, 3.05);
            let max_p = vec3<f32>(f32(dims.x) - 3.05, f32(dims.y) - 3.05, f32(dims.z) - 3.05);
            if next_pos.x < min_p.x { next_pos.x = min_p.x; vel.x = max(vel.x, 0.0); }
            if next_pos.x > max_p.x { next_pos.x = max_p.x; vel.x = min(vel.x, 0.0); }
            if next_pos.y < min_p.y { next_pos.y = min_p.y; vel.y = max(vel.y, 0.0); }
            if next_pos.y > max_p.y { next_pos.y = max_p.y; vel.y = min(vel.y, 0.0); }
            if next_pos.z < min_p.z { next_pos.z = min_p.z; vel.z = max(vel.z, 0.0); }
            if next_pos.z > max_p.z { next_pos.z = max_p.z; vel.z = min(vel.z, 0.0); }

            let speed = length(vel);
            let curl = length(vec3<f32>(c[1].z - c[2].y, c[2].x - c[0].z, c[0].y - c[1].x));
            let ratio = pred_dens / rest_density;
            let crest = smoothstep(0.92, 0.32, ratio);
            let foam_gen = (smoothstep(0.82, 1.90, speed) * 0.90 + smoothstep(1.15, 2.9, curl) * 0.45) * crest;
            let foam = clamp(prev_meta.x * 0.90 + foam_gen * dt * 2.1, 0.0, 1.0);
            let energy = clamp(prev_meta.y * 0.88 + (speed * 0.6 + curl * 0.35) * dt * 1.45, 0.0, 1.5);

            let center_sh = ((ur.z + 1u) << 6u) | ((ur.y + 1u) << 3u) | (ur.x + 1u);
            let m_xm = unpack_grid_vel(sh_vel_8[center_sh - 1u]).w;
            let m_xp = unpack_grid_vel(sh_vel_8[center_sh + 1u]).w;
            var m_ym = unpack_grid_vel(sh_vel_8[center_sh - 8u]).w;
            let m_yp = unpack_grid_vel(sh_vel_8[center_sh + 8u]).w;
            let m_zm = unpack_grid_vel(sh_vel_8[center_sh - 64u]).w;
            let m_zp = unpack_grid_vel(sh_vel_8[center_sh + 64u]).w;
            if cell.y + 1 <= 3 && m_yp > rest_density * 0.85 {
                m_ym = max(m_ym, rest_density);
            }
            let min_face = min(min(min(m_xm, m_xp), min(m_ym, m_yp)), min(m_zm, m_zp));
            let buried = min_face > rest_density * 0.78 && grid_dens > rest_density * 0.92;
            if !buried {
                if grid_dens > rest_density * 0.78 {
                    let shift = (center_sh & 3u) << 3u;
                    let old = (atomicAdd(&sh_cell_seen[center_sh >> 2u], 1u << shift) >> shift) & 255u;
                    let max_keep = select(2u, 1u, grid_dens > rest_density * 0.96);
                    emit_surface = old < max_keep;
                } else {
                    emit_surface = true;
                }
            }

            sim_particles[pid] = vec4<f32>(next_pos, pack_meta(foam, energy, ratio, speed));
            p2g_scatter_shared(next_pos, vel, c, pred_dens, a_orig, lid);
        }
    }

    var local_slot = 0xffffffffu;
    if emit_surface {
        local_slot = atomicAdd(&wg_surf_count, 1u);
    }
    workgroupBarrier();
    if lid == 0u {
        let count = atomicLoad(&wg_surf_count);
        if count > 0u {
            wg_surf_base = atomicAdd(&draw_args[1], count);
        }
    }
    flush_shared_atom(a_orig, lid);
    workgroupBarrier();
    if local_slot != 0xffffffffu {
        surface_indices[wg_surf_base + local_slot] = out_pid;
    }
}

@compute @workgroup_size(8, 8, 1)
fn build_heightfield(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = sim_params.dims.xyz;
    let gx = id.x;
    let gz = id.y;
    if gx >= dims.x || gz >= dims.z {
        return;
    }

    var top_y = 0.0;
    var min_y = f32(dims.y);
    var col_mass = 0.0;
    var top_speed = 0.0;
    var prev_m = 0.0;
    let thresh = 0.85;

    for (var iy = i32(dims.y) - 3; iy >= 2; iy -= 1) {
        let cell = unpack_grid_vel(grid_vel[cell_idx(gx, u32(iy), gz)]);
        let m = cell.w;
        col_mass += m;
        if m > thresh {
            let fy = f32(iy);
            if top_y <= 0.0 {
                let frac = clamp((m - thresh) / max(m - prev_m, 0.01), 0.0, 1.0);
                top_y = fy + frac;
                top_speed = length(cell.xyz);
            }
            min_y = min(min_y, fy);
        }
        prev_m = m;
    }

    if min_y >= f32(dims.y) {
        min_y = 0.0;
    }
    heightfield_out[gz * dims.x + gx] = vec4<f32>(top_y, col_mass, top_speed, min_y);
}
