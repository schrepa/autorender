import package::engine::view::{camera, stable_normalize, view_clip};
import package::effects::items::{EffectItem, effect_axes, effect_corner, effect_items, effect_lattice,
    effect_lit, effect_noise, effects_frame};
import package::effects::field::effect_field;

const SLAB_SKIN: u32 = 0u;
const SLAB_TOP: u32 = 1u;
const SLAB_END: u32 = 2u;
const SLAB_STRIPS: u32 = 30u;
const SLAB_CAP: u32 = 12u;

struct SlabCorner {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) local: vec3<f32>,
    @location(3) @interpolate(flat) index: u32,
    @location(4) @interpolate(flat) face: u32,
    @location(5) field: vec4<f32>,
}

fn slab_top(item: EffectItem, x: f32) -> f32 {
    let share = clamp(x / max(item.p0.w, 1.0e-3) * 0.5 + 0.5, 0.0, 1.0);
    return mix(item.p1.y, item.p1.z, share);
}

fn slab_profile(item: EffectItem, point: u32, x: f32) -> vec2<f32> {
    let thick = item.p1.x;
    let bevel = min(item.p1.w, thick * 0.9);
    let top = slab_top(item, x);
    let low = -item.p5.w;
    switch point {
        case 0u: { return vec2<f32>(thick, low); }
        case 1u: { return vec2<f32>(thick, top - bevel); }
        case 2u: { return vec2<f32>(thick - bevel, top); }
        case 3u: { return vec2<f32>(bevel - thick, top); }
        case 4u: { return vec2<f32>(-thick, top - bevel); }
        default: { return vec2<f32>(-thick, low); }
    }
}

@vertex fn vs_slab(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> SlabCorner {
    let item = effect_items[instance];
    let half = item.p0.w;
    let slope = (item.p1.z - item.p1.y) / max(2.0 * half, 1.0e-3);
    var local = vec3<f32>(0.0);
    var normal = vec3<f32>(0.0);
    var face = SLAB_END;
    if vertex < SLAB_STRIPS {
        let strip = vertex / 6u;
        let unit = effect_corner(vertex) * 0.5 + 0.5;
        let x = (unit.x * 2.0 - 1.0) * half;
        let lower = slab_profile(item, strip, x);
        let upper = slab_profile(item, strip + 1u, x);
        let at = mix(lower, upper, unit.y);
        local = vec3<f32>(x, at.y, at.x);
        let run = upper - lower;
        let side = normalize(vec2<f32>(run.y, -run.x) + vec2<f32>(1.0e-6, 0.0));
        normal = vec3<f32>(-slope * side.y, side.y, side.x);
        face = select(SLAB_TOP, SLAB_SKIN, strip == 0u || strip == 4u);
    } else {
        let cap = (vertex - SLAB_STRIPS) / SLAB_CAP;
        let corner = (vertex - SLAB_STRIPS) % SLAB_CAP;
        let fan = corner / 3u;
        let at_fan = corner % 3u;
        let point = select(fan + at_fan, 0u, at_fan == 0u);
        let x = select(-half, half, cap == 1u);
        let at = slab_profile(item, point, x);
        local = vec3<f32>(x, at.y, at.x);
        normal = vec3<f32>(select(-1.0, 1.0, cap == 1u), 0.0, 0.0);
    }
    let axes = effect_axes(item);
    let world = item.p0.xyz + axes * local;
    var out: SlabCorner;
    out.clip = view_clip(world);
    out.world = world;
    out.normal = axes * normal;
    out.local = local;
    out.index = instance;
    out.face = face;
    out.field = effect_field(world);
    return out;
}

fn rind_stripe_edge(x: f32, y: f32, seed: f32) -> f32 {
    let period = max(effects_frame.rind_skin.w, 0.02);
    let wobble = 0.3 * sin(y * 1.9 + seed) + 0.15 * sin(y * 5.3 + seed * 1.7);
    let u = x / period + wobble + seed;
    let lane = floor(u);
    let jag = effect_noise(vec3<f32>(lane * 3.1, y * 7.0, seed)) - 0.5;
    let width = 0.32 + 0.08 * sin(y * 3.7 + lane * 2.3) + 0.14 * jag;
    return width - abs(fract(u) - 0.5) * 2.0;
}

fn rind_seed_spot(plane: vec2<f32>, seed: f32, grain: f32) -> f32 {
    let spacing = max(effects_frame.rind_pith.w, 0.02);
    let cell = floor(plane / spacing);
    let jitter = vec2<f32>(effect_lattice(vec3<f32>(cell, seed + 1.0)),
        effect_lattice(vec3<f32>(cell, seed + 2.0)));
    let spot = (cell + 0.25 + 0.5 * jitter) * spacing;
    let reach = length((plane - spot) * vec2<f32>(1.0, 1.7));
    let size = spacing * 0.11;
    let kept = step(0.35, effect_lattice(vec3<f32>(cell, seed + 3.0)));
    return (1.0 - smoothstep(size - grain, size + grain, reach)) * kept;
}

fn rind_cracks(local: vec3<f32>, health: f32, grain: f32) -> f32 {
    let damage = 1.0 - saturate(health);
    let line = abs(effect_noise(local * 2.5) - 0.5);
    let width = damage * 0.05;
    return select(0.0, 1.0 - smoothstep(width, width + 0.01 + grain, line), damage > 0.0);
}

fn rind_glint(normal: vec3<f32>, view: vec3<f32>, power: f32) -> f32 {
    let sun = effects_frame.sun;
    let half_way = stable_normalize(sun.xyz + view);
    return select(0.0, pow(saturate(dot(normal, half_way)), power), sun.w != 0.0);
}

@fragment fn fs_slab(corner: SlabCorner) -> @location(0) vec4<f32> {
    let item = effect_items[corner.index];
    let local = corner.local;
    let seed = item.p2.x;
    let half = item.p0.w;
    let grain = max(length(fwidth(local)), 1.0e-4);
    let period = max(effects_frame.rind_skin.w, 0.02);
    let edge = rind_stripe_edge(local.x, local.y, seed);
    let soft = max(fwidth(edge), 1.0e-4);
    let far = saturate(grain / period * 3.0 - 0.5);
    let striped = mix(smoothstep(-soft, soft, edge), 0.3, far);
    let speckle = mix(0.88 + 0.24 * effect_noise(local * 22.0 + vec3<f32>(seed)), 1.0,
        smoothstep(0.015, 0.05, grain));
    let cracked = rind_cracks(local, item.p2.y, grain);
    var skin = mix(effects_frame.rind_skin.rgb, effects_frame.rind_stripe.rgb, striped) * speckle;
    skin = mix(skin, effects_frame.rind_flesh.rgb * 0.5, cracked);

    let across = item.p1.x - abs(local.z);
    let skin_m = effects_frame.rind_flesh.w;
    let pith_m = effects_frame.rind_stripe.w;
    let fleshy = smoothstep(skin_m + pith_m * 0.6 - grain, skin_m + pith_m + grain, across);
    let plane = select(local.xz, local.yz, corner.face == SLAB_END);
    let seeded = rind_seed_spot(plane, seed, grain) * smoothstep(0.6, 1.0, fleshy)
        * (1.0 - smoothstep(0.02, 0.06, grain));
    let fibre = mix(0.85 + 0.3 * effect_noise(vec3<f32>(plane * 30.0, seed)), 1.0,
        smoothstep(0.01, 0.03, grain));
    var cut = mix(effects_frame.rind_skin.rgb, effects_frame.rind_pith.rgb,
        smoothstep(skin_m - grain, skin_m + grain, across));
    cut = mix(cut, effects_frame.rind_flesh.rgb * fibre, fleshy);
    cut = mix(cut, effects_frame.rind_seed.rgb, seeded);

    let rounded = effects_frame.rind_seed.w * local.x / max(half, 1.0e-3);
    let flat_normal = stable_normalize(corner.normal);
    let is_skin = corner.face == SLAB_SKIN;
    let normal = select(flat_normal, stable_normalize(flat_normal + item.p4.xyz * rounded), is_skin);
    let view = stable_normalize(camera.position.xyz - corner.world);
    let ground = local.y - item.p2.z;
    let seam = select(1.0, mix(0.55, 1.0, smoothstep(0.0, 0.08, half - abs(local.x))), is_skin);
    let occlusion = seam * mix(0.45, 1.0, smoothstep(0.0, 0.6, ground));
    let base = select(cut, skin, is_skin) * occlusion;
    var shown = effect_lit(base, normal, corner.field);
    let wet = select(0.45 * fleshy, 0.22, is_skin);
    let glint = rind_glint(normal, view, select(60.0, 36.0, is_skin)) * wet * corner.field.a;
    shown += effects_frame.sun_color.rgb * glint;
    let rim = pow(1.0 - abs(dot(normal, view)), effects_frame.panel.z) * effects_frame.panel.y;
    shown += effects_frame.sun_color.rgb * rim * occlusion;
    let below_top = slab_top(item, local.x) - item.p1.w - local.y;
    let trim_m = effects_frame.panel.x;
    let trim = select(0.0, 1.0 - smoothstep(trim_m - grain, trim_m + grain, below_top),
        is_skin && below_top >= 0.0);
    return vec4<f32>(mix(shown, item.p3.rgb * effects_frame.panel.w, trim), 1.0);
}
