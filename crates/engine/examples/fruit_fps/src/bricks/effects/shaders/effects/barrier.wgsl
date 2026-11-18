import package::engine::view::view_clip;
import package::effects::items::{effect_corner, effect_items, effect_lattice, effect_seconds};

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

@fragment fn fs_barrier(corner: BarrierCorner) -> @location(0) vec4<f32> {
    let item = effect_items[corner.index];
    let size = vec2<f32>(item.p0.w, item.p1.w);
    let local = corner.local;
    let pattern = item.p5;
    let alphas = item.p6;
    let drop = item.p2.w;
    let blur = max(fwidth(local), vec2<f32>(1.0e-4)) * pattern.w;
    let to_line = abs(fract(local / pattern.x + 0.5) - 0.5) * pattern.x;
    let lines = 1.0 - smoothstep(vec2<f32>(0.0), blur, to_line);
    let grid = max(lines.x, lines.y);
    let phase = (local.x + local.y - effect_seconds() * pattern.z) / pattern.y;
    let band = smoothstep(0.3, 0.5, abs(fract(phase) - 0.5));
    let to_edge = min(min(local.x, size.x - local.x), min(local.y, size.y - local.y));
    let edge = 1.0 - smoothstep(0.0, alphas.w, to_edge);
    let near = item.p4;
    let close = select(0.0, 1.0 - smoothstep(0.0, near.w, distance(corner.world, near.xyz)),
        near.w > 0.0);
    let chance = effect_lattice(vec3<f32>(floor(local / pattern.x), item.p7.x));
    let rim = select(0.0, 1.0 - smoothstep(0.0, 0.15, chance - drop), drop > 0.0);
    let strength = alphas.x + alphas.y * max(grid, band * item.p7.y) + alphas.z * edge;
    let alpha = saturate(strength * (1.0 + item.p3.w * close) + rim * alphas.z);
    if chance < drop || alpha < 0.004 {
        discard;
    }
    return vec4<f32>(item.p3.rgb * (1.0 + rim) * alpha, alpha);
}
