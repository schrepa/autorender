import package::engine::view::view_clip;
import package::generated::world::WorldUniforms;
import package::generated::world::Particle;

@group(2) @binding(0) var<uniform> world: WorldUniforms;
@group(2) @binding(1) var<storage, read> particles: array<Particle>;

struct PartOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) kind_spin: vec2<f32>,
}

@vertex
fn vs_particle(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> PartOut {
    var out: PartOut;
    let p = particles[iid].pos_phase;
    let v = particles[iid].vel_kind;
    let pos = p.xyz;
    let phase = p.w;
    let is_petal = v.w >= 0.9;
    let t = world.time_wind.x;

    let dist = length(pos - world.camera_pos.xyz);
    let max_d = select(46.0, 34.0, is_petal);
    if dist < 1.35 || dist > max_d {
        out.clip_pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vid];

    let near_fade = smoothstep(1.35, 3.8, dist);
    var size = 0.0;
    var col = vec4<f32>(0.0);
    if !is_petal {
        let ff_power = world.sky_horizon.w;
        if ff_power < 0.05 {
            out.clip_pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
            return out;
        }
        let is_hero = iid < 28u;
        let wave = sin(t * (1.25 + fract(v.w) * 1.15) + phase * 5.3);
        let blink_exp = select(6.2, 1.8, is_hero);
        let pulse = pow(max(wave, 0.0), blink_exp);
        let gate = select(0.11, 0.02, is_hero);
        if pulse < gate {
            out.clip_pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
            return out;
        }
        let hero_boost = select(1.0, 1.65, is_hero);
        size = (0.022 + 0.028 * pulse) * hero_boost * (1.0 + smoothstep(14.0, 42.0, dist) * 0.55);
        let tint = mix(vec3<f32>(0.68, 1.0, 0.20), vec3<f32>(1.0, 0.72, 0.16), fract(phase * 3.1));
        let alpha = pulse * min(ff_power, 1.0) * near_fade * (1.0 - smoothstep(32.0, 46.0, dist));
        col = vec4<f32>(tint * (2.6 + 2.8 * ff_power * pulse), alpha);
    } else {
        let petal_fade = smoothstep(2.2, 5.2, dist);
        if petal_fade < 0.02 || ((iid & 3u) != 0u && dist > 18.0) {
            out.clip_pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
            return out;
        }
        let spin = t * (2.6 + fract(v.w) * 2.2) + phase * 5.0;
        let tilt = 0.36 + 0.64 * abs(cos(spin * 0.7));
        size = (0.026 + 0.018 * fract(phase * 5.1)) * (1.0 + smoothstep(12.0, 32.0, dist) * 0.35);
        let day_lit = mix(vec3<f32>(0.98, 0.58, 0.68), vec3<f32>(1.0, 0.84, 0.84), fract(phase * 2.3));
        let sun_boost = world.sun_color.rgb * 0.34 + world.sky_horizon.rgb * 0.40 + vec3<f32>(0.12, 0.10, 0.14);
        let fade = petal_fade * (1.0 - smoothstep(22.0, 34.0, dist)) * tilt;
        col = vec4<f32>(day_lit * sun_boost, 0.82 * fade);
    }

    let spin_ang = select(0.0, t * 2.4 + phase * 6.0, is_petal);
    let cs = cos(spin_ang);
    let sn = sin(spin_ang);
    let rot_c = vec2<f32>(corner.x * cs - corner.y * sn, corner.x * sn + corner.y * cs);
    let offset = (world.camera_right.xyz * rot_c.x + world.camera_up.xyz * rot_c.y) * size;

    out.clip_pos = view_clip(pos + offset);
    out.uv = corner;
    out.color = col;
    out.kind_spin = vec2<f32>(select(0.0, 1.0, is_petal), phase);
    return out;
}

@fragment
fn fs_particle(in: PartOut) -> @location(0) vec4<f32> {
    let d2 = dot(in.uv, in.uv);
    if d2 > 1.0 {
        discard;
    }
    if in.kind_spin.x < 0.5 {
        let core = exp(-d2 * 12.0);
        let halo = exp(-d2 * 3.6) * 0.42;
        let shape = core + halo;
        return vec4<f32>(in.color.rgb * (1.0 + core * 1.2), clamp(shape * in.color.a, 0.0, 1.0));
    }
    let p = vec2<f32>(in.uv.x * 1.15, in.uv.y * 0.82);
    let notch = smoothstep(0.0, 0.25, abs(p.x)) * 0.18 * step(0.0, p.y);
    let petal = 1.0 - smoothstep(0.55, 0.92, length(p) + notch);
    if petal < 0.05 {
        discard;
    }
    return vec4<f32>(in.color.rgb, petal * in.color.a);
}
