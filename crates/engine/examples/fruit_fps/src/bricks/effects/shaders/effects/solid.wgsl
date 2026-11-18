import package::engine::view::{stable_normalize, view_clip};
import package::effects::items::{EFFECT_TAU, effect_axes, effect_box_corner, effect_items, effect_lit,
    effect_seconds};
import package::effects::field::effect_field;

struct SolidCorner {
    @builtin(position) clip: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) index: u32,
    @location(3) local: vec3<f32>,
    @location(4) @interpolate(flat) field: vec4<f32>,
}

@vertex fn vs_solid(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> SolidCorner {
    let item = effect_items[instance];
    let corner = effect_box_corner(vertex);
    let axes = effect_axes(item);
    let world = item.p0.xyz + axes * (corner.local * item.p1.xyz);
    var out: SolidCorner;
    out.clip = view_clip(world);
    out.normal = axes * corner.normal;
    out.index = instance;
    out.local = corner.local;
    out.field = effect_field(item.p0.xyz);
    return out;
}

@fragment fn fs_solid(corner: SolidCorner) -> @location(0) vec4<f32> {
    let item = effect_items[corner.index];
    let kind = u32(item.p3.x);
    let normal = stable_normalize(corner.normal);
    let sky = 0.55 + 0.45 * normal.y;
    if kind == 1u {
        let wave = 0.5 + 0.5 * sin(effect_seconds() * item.p1.w * EFFECT_TAU);
        let beat = select(1.0, 0.6 + 0.4 * wave, item.p3.z > 0.0);
        return vec4<f32>(item.p2.rgb * item.p2.w * beat * (0.8 + 0.2 * sky), 1.0);
    }
    if kind == 3u {
        let share = item.p3.y;
        let dither = fract(52.9829189 * fract(dot(corner.clip.xy, vec2<f32>(0.06711056, 0.00583715))));
        if dither >= share {
            discard;
        }
        let wave = 0.5 + 0.5 * sin(effect_seconds() * item.p1.w * EFFECT_TAU);
        let beat = select(1.0, 0.6 + 0.4 * wave, item.p3.z > 0.0);
        let vine = effect_lit(item.p6.rgb, normal, corner.field);
        let period = max(item.p7.z, 0.1);
        let along = corner.local.z * item.p1.z - effect_seconds() * item.p7.y;
        let travel = 1.0 - smoothstep(0.0, 0.25, abs(fract(along / period) - 0.5) * period);
        let glow = item.p2.rgb * item.p2.w * (beat + 2.0 * travel) * share;
        return vec4<f32>(vine + glow, 1.0);
    }
    return vec4<f32>(item.p2.rgb * sky, 1.0);
}
