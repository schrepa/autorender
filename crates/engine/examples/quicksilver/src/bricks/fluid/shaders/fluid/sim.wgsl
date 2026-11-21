import package::fluid::common::{
    Particle, SimParams, cell_idx, decode_fixed, encode_fixed, pack_c_vec, rand01, unpack_c,
};

@group(1) @binding(0) var<uniform> sim_params: SimParams;
@group(1) @binding(1) var<storage, read_write> sim_particles: array<Particle>;
@group(1) @binding(2) var<storage, read_write> grid_atomic: array<atomic<i32>>;
@group(1) @binding(3) var<storage, read_write> grid_vel: array<vec4<f32>>;
@group(1) @binding(4) var<storage, read_write> heightfield_out: array<vec4<f32>>;

fn bspline_weights(fx: vec3<f32>) -> array<vec3<f32>, 3> {
    let w0 = 0.5 * (1.5 - fx) * (1.5 - fx);
    let w1 = 0.75 - (fx - 1.0) * (fx - 1.0);
    let w2 = 0.5 * (fx - 0.5) * (fx - 0.5);
    return array<vec3<f32>, 3>(w0, w1, w2);
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

    sim_particles[i] = Particle(
        vec4<f32>(pos, 0.0),
        vec4<f32>(vel, rest_d),
        vec4<u32>(0u),
        vec4<f32>(0.0, 0.0, 1.0, 1.0),
    );
}

@compute @workgroup_size(128)
fn drop_burst(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x;
    let drop_count = sim_params.drop_info.y;
    if k >= drop_count {
        return;
    }
    let count = sim_params.dims.w;
    let i = (sim_params.drop_info.x + k * 7u) % count;
    let seed = sim_params.drop_info.z;
    let gx = f32(sim_params.dims.x);
    let gy = f32(sim_params.dims.y);
    let gz = f32(sim_params.dims.z);
    let rest_d = sim_params.physics.z;

    let cx = gx * (0.36 + 0.28 * rand01(seed + 3u));
    let cz = gz * (0.36 + 0.28 * rand01(seed + 11u));
    let cy = 18.5;

    let r0 = rand01(k * 3571u + seed);
    let r1 = rand01(k * 7919u + seed + 19u);
    let r2 = rand01(k * 14591u + seed + 47u);
    let u = r0 * 6.2831853;
    let cos_t = r1 * 2.0 - 1.0;
    let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
    let rad = 7.4 * pow(r2, 0.3333333);
    let dir = vec3<f32>(sin_t * cos(u), cos_t, sin_t * sin(u));

    let pos = clamp(
        vec3<f32>(cx, cy, cz) + dir * rad,
        vec3<f32>(4.0),
        vec3<f32>(gx - 4.5, gy - 4.0, gz - 4.5),
    );
    let vel = vec3<f32>((rand01(seed + 29u) - 0.5) * 0.30, -2.25, (rand01(seed + 37u) - 0.5) * 0.22);

    sim_particles[i] = Particle(
        vec4<f32>(pos, 0.0),
        vec4<f32>(vel, rest_d),
        vec4<u32>(0u),
        vec4<f32>(0.0, 0.35, 1.0, 1.0),
    );
}

@compute @workgroup_size(256)
fn clear_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total = sim_params.dims.x * sim_params.dims.y * sim_params.dims.z;
    if idx >= total {
        return;
    }
    let base = idx * 4u;
    atomicStore(&grid_atomic[base], 0);
    atomicStore(&grid_atomic[base + 1u], 0);
    atomicStore(&grid_atomic[base + 2u], 0);
    atomicStore(&grid_atomic[base + 3u], 0);
    grid_vel[idx] = vec4<f32>(0.0);
}

@compute @workgroup_size(128)
fn p2g_fused(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= sim_params.dims.w {
        return;
    }
    let p = sim_particles[i];
    let pos = p.pos_foam.xyz;
    let vel = p.vel_dens.xyz;
    let density = max(p.vel_dens.w, 0.35);
    let c = unpack_c(p.c_packed, p.extra.x);

    let dt = sim_params.physics.x;
    let mass = sim_params.physics.y;
    let rest_density = sim_params.physics.z;
    let stiffness = sim_params.physics.w;
    let viscosity = sim_params.fluid.x;
    let dims = sim_params.dims.xyz;

    let ratio = clamp(density / rest_density, 0.2, 2.6);
    let eq_p = stiffness * (ratio * ratio * ratio - 1.0);
    let pressure = clamp(eq_p, -0.06, 85.0);
    let volume = mass / density;

    let strain = c + transpose(c);
    let stress = mat3x3<f32>(
        vec3<f32>(-pressure, 0.0, 0.0),
        vec3<f32>(0.0, -pressure, 0.0),
        vec3<f32>(0.0, 0.0, -pressure),
    ) + viscosity * strain;
    let affine = mass * c - (volume * 4.0 * dt) * stress;
    let mv = mass * vel;

    let cell = vec3<i32>(floor(pos - 0.5));
    let fx = pos - vec3<f32>(cell);
    let w = bspline_weights(fx);

    for (var oz = 0; oz < 3; oz += 1) {
        let gz = cell.z + oz;
        if gz < 0 || gz >= i32(dims.z) {
            continue;
        }
        let wz = w[oz].z;
        for (var oy = 0; oy < 3; oy += 1) {
            let gy = cell.y + oy;
            if gy < 0 || gy >= i32(dims.y) {
                continue;
            }
            let wyz = w[oy].y * wz;
            for (var ox = 0; ox < 3; ox += 1) {
                let gx = cell.x + ox;
                if gx < 0 || gx >= i32(dims.x) {
                    continue;
                }
                let weight = w[ox].x * wyz;
                if weight < 0.0005 {
                    continue;
                }
                let dpos = vec3<f32>(f32(ox), f32(oy), f32(oz)) - fx;
                let mw = mass * weight;
                let mom = (mv + affine * dpos) * weight;
                let base = cell_idx(u32(gx), u32(gy), u32(gz), dims) * 4u;
                let mx = encode_fixed(mom.x);
                let my = encode_fixed(mom.y);
                let mz = encode_fixed(mom.z);
                let mm = encode_fixed(mw);
                if mx != 0 { atomicAdd(&grid_atomic[base], mx); }
                if my != 0 { atomicAdd(&grid_atomic[base + 1u], my); }
                if mz != 0 { atomicAdd(&grid_atomic[base + 2u], mz); }
                if mm != 0 { atomicAdd(&grid_atomic[base + 3u], mm); }
            }
        }
    }
}

@compute @workgroup_size(256)
fn update_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let dims = sim_params.dims.xyz;
    let total = dims.x * dims.y * dims.z;
    if idx >= total {
        return;
    }
    let base = idx * 4u;
    let ax = atomicLoad(&grid_atomic[base]);
    let ay = atomicLoad(&grid_atomic[base + 1u]);
    let az = atomicLoad(&grid_atomic[base + 2u]);
    let am = atomicLoad(&grid_atomic[base + 3u]);
    if (ax | ay | az | am) != 0 {
        atomicStore(&grid_atomic[base], 0);
        atomicStore(&grid_atomic[base + 1u], 0);
        atomicStore(&grid_atomic[base + 2u], 0);
        atomicStore(&grid_atomic[base + 3u], 0);
    }

    let mass = decode_fixed(am);
    if mass <= 1.0e-5 {
        grid_vel[idx] = vec4<f32>(0.0);
        return;
    }

    let gx = idx % dims.x;
    let gy = (idx / dims.x) % dims.y;
    let gz = idx / (dims.x * dims.y);
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

    grid_vel[idx] = vec4<f32>(vel, mass);
}

@compute @workgroup_size(128)
fn g2p(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= sim_params.dims.w {
        return;
    }
    let p = sim_particles[i];
    let pos = p.pos_foam.xyz;
    let dt = sim_params.physics.x;
    let rest_density = sim_params.physics.z;
    let dims = sim_params.dims.xyz;

    let cell = vec3<i32>(floor(pos - 0.5));
    let fx = pos - vec3<f32>(cell);
    let w = bspline_weights(fx);

    var vel = vec3<f32>(0.0);
    var b0 = vec3<f32>(0.0);
    var b1 = vec3<f32>(0.0);
    var b2 = vec3<f32>(0.0);
    var grid_dens = 0.0;
    var min_face_m = 1.0e9;

    for (var oz = 0; oz < 3; oz += 1) {
        let gz = cell.z + oz;
        if gz < 0 || gz >= i32(dims.z) {
            min_face_m = 0.0;
            continue;
        }
        let wz = w[oz].z;
        for (var oy = 0; oy < 3; oy += 1) {
            let gy = cell.y + oy;
            if gy < 0 || gy >= i32(dims.y) {
                min_face_m = 0.0;
                continue;
            }
            let wyz = w[oy].y * wz;
            for (var ox = 0; ox < 3; ox += 1) {
                let gx = cell.x + ox;
                if gx < 0 || gx >= i32(dims.x) {
                    min_face_m = 0.0;
                    continue;
                }
                let weight = w[ox].x * wyz;
                let gcell = grid_vel[cell_idx(u32(gx), u32(gy), u32(gz), dims)];
                let dpos = vec3<f32>(f32(ox), f32(oy), f32(oz)) - fx;
                let wv = gcell.xyz * weight;
                vel += wv;
                b0 += wv * dpos.x;
                b1 += wv * dpos.y;
                b2 += wv * dpos.z;
                grid_dens += gcell.w * weight;
                let manh = abs(ox - 1) + abs(oy - 1) + abs(oz - 1);
                if manh == 1 {
                    min_face_m = min(min_face_m, gcell.w);
                }
            }
        }
    }

    let c = mat3x3<f32>(b0 * 4.0, b1 * 4.0, b2 * 4.0);
    let div_v = clamp(c[0].x + c[1].y + c[2].z, -3.2, 3.2);
    let pred_dens = clamp(grid_dens / clamp(1.0 + dt * div_v, 0.72, 1.38), 0.3, rest_density * 2.8);
    let ratio = pred_dens / rest_density;

    var next_pos = pos + vel * dt;
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
    let crest = smoothstep(0.92, 0.32, ratio);
    let foam_gen = (smoothstep(0.82, 1.90, speed) * 0.90 + smoothstep(1.15, 2.9, curl) * 0.45) * crest;
    let foam = clamp(p.pos_foam.w * 0.94 + foam_gen * dt * 1.25, 0.0, 1.0);
    let energy = clamp(p.extra.y * 0.93 + (speed * 0.6 + curl * 0.35) * dt * 0.85, 0.0, 1.5);
    let buried = min_face_m > rest_density * 0.78 && grid_dens > rest_density * 0.92;
    let draw_surface = select(1.0, 0.0, buried && (i & 3u) != 0u);

    sim_particles[i] = Particle(
        vec4<f32>(next_pos, foam),
        vec4<f32>(vel, pred_dens),
        pack_c_vec(c),
        vec4<f32>(clamp(c[2].z, -40.0, 40.0), energy, draw_surface, ratio),
    );
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
        let cell = grid_vel[cell_idx(gx, u32(iy), gz, dims)];
        let m = cell.w;
        col_mass += m;
        if m > thresh {
            let fy = f32(iy);
            if top_y <= 0.0 {
                let frac = clamp((m - thresh) / max(m - prev_m, 0.01), 0.0, 1.0);
                top_y = fy + (1.0 - frac);
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
