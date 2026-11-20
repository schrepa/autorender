import package::engine::view::{camera, view_clip};
import package::effects::items::{EFFECT_TAU, effect_corner, effect_items, effect_lattice, effect_seconds,
    effects_frame};

struct BarrierCorner {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) world: vec3<f32>,
    @location(2) @interpolate(flat) index: u32,
}

@vertex fn vs_barrier(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> BarrierCorner {
    let item = effect_items[instance];
    let size = vec2<f32>(item.p0.w, item.p1.w);
    let local = (effect_corner(vertex) * 0.5 + 0.5) * size;
    let world = item.p0.xyz + item.p1.xyz * local.x + item.p2.xyz * local.y;
    var out: BarrierCorner;
    out.clip = view_clip(world);
    out.local = local;
    out.world = world;
    out.index = instance;
    return out;
}

const BARRIER_SPACING: vec2<f32> = vec2<f32>(1.0, 1.7320508);
const BARRIER_INV_SPACING: vec2<f32> = vec2<f32>(1.0, 0.57735027);
const BARRIER_MIDDLE: vec2<f32> = vec2<f32>(0.5, 0.8660254);

fn barrier_hex(point: vec2<f32>) -> vec3<f32> {
    let a = point - BARRIER_SPACING * floor(point * BARRIER_INV_SPACING) - BARRIER_MIDDLE;
    let shifted = point - BARRIER_MIDDLE;
    let b = shifted - BARRIER_SPACING * floor(shifted * BARRIER_INV_SPACING) - BARRIER_MIDDLE;
    let offset = select(b, a, dot(a, a) < dot(b, b));
    let away = abs(offset);
    let reach = max(away.x * 0.5 + away.y * 0.8660254, away.x);
    return vec3<f32>(0.5 - reach, point - offset);
}

@fragment fn fs_barrier(corner: BarrierCorner) -> @location(0) vec4<f32> {
    let item = effect_items[corner.index];
    let size = vec2<f32>(item.p0.w, item.p1.w);
    let local = corner.local;
    let pattern = item.p5;
    let alphas = item.p6;
    let shape = item.p7;
    let look = effects_frame.barrier;
    let seconds = effect_seconds();
    let drop = item.p2.w;
    let cell = max(pattern.x, 0.05);
    let flowing = vec2<f32>(local.x, local.y - seconds * look.z) / cell;
    let grain = max(length(fwidth(flowing)), 1.0e-4);
    let hex = barrier_hex(flowing);
    let width = grain * look.w;
    let lines = (1.0 - smoothstep(width * 0.5, width * 1.5, hex.x)) * (1.0 - smoothstep(0.12, 0.35, grain));
    var twinkle = 0.0;
    var flare = 0.0;
    if lines > 0.0 || drop > 0.0 {
        let chance = effect_lattice(vec3<f32>(hex.yz, shape.x));
        if drop > 0.0 {
            if chance < drop {
                discard;
            }
            flare = 1.0 - smoothstep(0.0, 0.15, chance - drop);
        }
        if lines > 0.0 {
            twinkle = 0.5 + 0.5 * sin(seconds * 0.9 + chance * EFFECT_TAU);
        }
    }
    let phase = (local.x + local.y * 0.6 - seconds * pattern.z) / max(pattern.y, 0.1);
    let wave = 0.5 + 0.5 * cos(phase * EFFECT_TAU);
    let wave_2 = wave * wave;
    let band = wave_2 * wave_2;
    let rise = saturate(local.y / max(size.y, 0.01));
    let fading = 1.0 - smoothstep(shape.z, 1.0, rise);
    var ground = 0.0;
    if alphas.z > 0.0 {
        let decay = max(alphas.w, 0.01);
        if local.y < decay * 7.0 {
            ground = alphas.z * exp(-local.y / decay);
        }
    }
    let side = smoothstep(0.0, max(shape.y, 0.01), min(local.x, size.x - local.x));
    let near = item.p4;
    var close = 0.0;
    if near.w > 0.0 {
        close = 1.0 - smoothstep(0.0, near.w, distance(corner.world, near.xyz));
    }
    let eye = mix(look.y, 1.0, smoothstep(0.0, max(look.x, 0.01), distance(corner.world, camera.position.xyz)));
    let cells = alphas.x + alphas.y * lines * (0.55 + 0.45 * twinkle) * (1.0 + pattern.w * band);
    let glow = alphas.x * pattern.w * band;
    let strength = ((cells + glow) * fading + ground) * (1.0 + item.p3.w * close);
    let alpha = saturate((strength + flare * alphas.y) * side * eye);
    if alpha < 0.004 {
        discard;
    }
    return vec4<f32>(item.p3.rgb * (1.0 + flare) * alpha, alpha);
}
