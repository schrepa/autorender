import package::engine::view::{camera, stable_normalize, view_clip};
import package::effects::items::{EFFECT_TAU, EffectItem, effect_axes, effect_box_corner,
    effect_items, effect_lit, effect_noise, effect_random, effects_frame, effect_seconds};
import package::effects::field::effect_field;

struct SolidCorner {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
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
    out.world = world;
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
    let view = stable_normalize(camera.position.xyz - corner.world);
    let sky = 0.55 + 0.45 * normal.y;
    if kind == 4u || kind == 5u {
        return vec4<f32>(panel_shade(item, corner, normal, view), 1.0);
    }
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

struct Panel {
    metres: vec3<f32>,
    half: vec3<f32>,
    face: u32,
    trim: f32,
}

fn panel_of(item: EffectItem, local: vec3<f32>) -> Panel {
    let half = item.p1.xyz;
    let metres = local * half;
    let away = abs(local);
    var face = 2u;
    if away.y >= away.x && away.y >= away.z {
        face = 1u;
    } else if away.x >= away.z {
        face = 0u;
    }
    let from_edge = min(half.x - abs(metres.x), half.y - metres.y);
    let trim = 1.0 - smoothstep(0.0, effects_frame.panel.x, from_edge);
    return Panel(metres, half, face, select(trim, 0.0, face != 2u));
}

fn rind_section(across: f32, metres: vec3<f32>) -> vec3<f32> {
    let skin = 0.02;
    let pith = effects_frame.rind_stripe.w;
    if across < skin {
        return effects_frame.rind_skin.rgb;
    }
    if across < skin + pith {
        return mix(effects_frame.rind_pith.rgb, effects_frame.rind_flesh.rgb,
            smoothstep(skin + pith * 0.6, skin + pith, across));
    }
    let spacing = max(effects_frame.rind_pith.w, 0.02);
    let cell = floor(metres.xz / spacing);
    let seed = u32(i32(cell.x) * 7919 + i32(cell.y) * 104729 + 1000000);
    let spot = (vec2<f32>(effect_random(seed), effect_random(seed + 1u)) * 0.6 + 0.2 + cell) * spacing;
    let seeded = 1.0 - smoothstep(spacing * 0.12, spacing * 0.18, length(metres.xz - spot));
    return mix(effects_frame.rind_flesh.rgb, effects_frame.rind_seed.rgb, seeded);
}

fn rind_color(panel: Panel) -> vec3<f32> {
    let metres = panel.metres;
    if panel.face == 2u {
        let period = max(effects_frame.rind_skin.w, 0.05);
        let wave = metres.x + 0.12 * sin(metres.y * 2.3) + 0.08 * effect_noise(metres * 4.0);
        let stripe = smoothstep(0.3, 0.45, abs(fract(wave / period) - 0.5));
        let speckle = 0.85 + 0.3 * effect_noise(metres * 18.0);
        return mix(effects_frame.rind_skin.rgb, effects_frame.rind_stripe.rgb, stripe) * speckle;
    }
    let across = panel.half.z - abs(metres.z);
    return rind_section(across, metres);
}

fn curtain_color(panel: Panel) -> vec3<f32> {
    let metres = panel.metres;
    let pleat = 0.5 + 0.5 * cos(metres.x / max(effects_frame.panel.z, 0.05) * EFFECT_TAU);
    let cell_m = max(effects_frame.curtain_peel.w, 0.03);
    let cells = effect_noise(metres / cell_m + vec3<f32>(0.0, effect_seconds() * 0.05, 0.0));
    let walls = smoothstep(0.05, 0.25, abs(cells - 0.5));
    let peel = mix(effects_frame.curtain_peel.rgb * 0.7, effects_frame.curtain_peel.rgb, pleat);
    let pulp = effects_frame.curtain_pulp.rgb * (0.7 + 0.3 * walls);
    return select(mix(peel, pulp, 0.55 * walls), effects_frame.curtain_peel.rgb, panel.face != 2u);
}

fn panel_cracks(panel: Panel, health: f32) -> f32 {
    let damage = 1.0 - saturate(health);
    if damage <= 0.0 {
        return 0.0;
    }
    let line = abs(effect_noise(panel.metres * 2.5) - 0.5);
    return 1.0 - smoothstep(damage * 0.05, damage * 0.05 + 0.01, line);
}

fn panel_shade(item: EffectItem, corner: SolidCorner, normal: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
    let panel = panel_of(item, corner.local);
    let rind = u32(item.p3.x) == 4u;
    var color = select(curtain_color(panel), rind_color(panel), rind);
    let cracked = panel_cracks(panel, item.p3.w);
    color = mix(color, select(effects_frame.curtain_pulp.rgb, effects_frame.rind_flesh.rgb, rind) * 0.5, cracked);
    var shown = effect_lit(color, normal, corner.field);
    if !rind {
        shown += effects_frame.curtain_pulp.rgb * effects_frame.curtain_pulp.w * (1.0 - cracked);
    }
    let rim = pow(1.0 - abs(dot(normal, view)), item.p6.w) * effects_frame.panel.y;
    shown += effects_frame.sun_color.rgb * rim;
    let team = item.p6.rgb * item.p7.x;
    return mix(shown, team, panel.trim);
}
