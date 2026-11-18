import package::engine::view::{camera, stable_normalize, view_clip};
import package::effects::items::{EFFECT_HIDDEN, EffectItem, effect_axes, effect_box_corner, effect_items,
    effect_lattice, effect_lit, effect_noise, effects_frame, effect_seconds};
import package::effects::field::effect_field;

struct SheetCorner {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) local: vec3<f32>,
    @location(2) @interpolate(flat) normal: vec3<f32>,
    @location(3) @interpolate(flat) index: u32,
    @location(4) field: vec4<f32>,
}

fn sheet_half(item: EffectItem) -> vec3<f32> {
    return vec3<f32>(item.p0.w + item.p1.z, (item.p1.y + item.p1.w) * 0.5, item.p1.x);
}

@vertex fn vs_sheet(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> SheetCorner {
    let item = effect_items[instance];
    let corner = effect_box_corner(vertex);
    let half = sheet_half(item);
    let axes = effect_axes(item);
    let centre = item.p0.xyz + item.p5.xyz * half.y;
    let eye = (camera.position.xyz - centre) * axes;
    let inside = all(abs(eye) < half);
    let facing = dot(corner.normal, eye - corner.normal * half) > 0.0;
    var out: SheetCorner;
    out.index = instance;
    out.world = vec3<f32>(0.0);
    out.local = vec3<f32>(0.0);
    out.normal = vec3<f32>(0.0, 1.0, 0.0);
    out.field = vec4<f32>(0.0);
    if !(inside || facing) {
        out.clip = EFFECT_HIDDEN;
        return out;
    }
    let world = centre + axes * (corner.local * half);
    out.clip = view_clip(world);
    out.world = world;
    out.local = corner.local * half + vec3<f32>(0.0, half.y, 0.0);
    out.normal = axes * select(corner.normal, -corner.normal, inside);
    out.field = effect_field(world);
    return out;
}

fn sheet_streaks(along: f32, flow: f32, scale: f32, seed: f32) -> f32 {
    return effect_noise(vec3<f32>(along / scale, flow / (scale * 8.0), seed));
}

fn sheet_fleck(plane: vec2<f32>, cell: f32, seed: f32, grain: f32) -> f32 {
    let at = plane / cell;
    let id = floor(at);
    let part = at - id;
    let spot = vec2<f32>(0.2 + 0.6 * effect_lattice(vec3<f32>(id, seed)),
        0.2 + 0.6 * effect_lattice(vec3<f32>(id, seed + 1.0)));
    let size = 0.12 + 0.12 * effect_lattice(vec3<f32>(id, seed + 2.0));
    let reach = length((part - spot) * vec2<f32>(1.0, 0.4));
    let soft = grain / cell;
    let kept = step(0.55, effect_lattice(vec3<f32>(id, seed + 3.0)))
        * (1.0 - smoothstep(0.25, 0.6, soft / size));
    return (1.0 - smoothstep(size - soft, size + soft, reach)) * kept;
}

fn sheet_drops(along: f32, up: f32, seed: f32, grain: f32) -> f32 {
    let shape = effects_frame.curtain_shape;
    let cell = max(shape.y * 0.4, 0.04);
    let lane = floor(along / cell);
    let roll = effect_lattice(vec3<f32>(lane, 0.0, seed));
    let phase = fract(effect_seconds() * (1.2 + roll) + roll * 7.0);
    let hop = shape.w * (0.35 + 0.65 * effect_lattice(vec3<f32>(lane, 1.0, seed)));
    let lift = 4.0 * phase * (1.0 - phase) * hop;
    let across = (lane + 0.2 + 0.6 * effect_lattice(vec3<f32>(lane, 2.0, seed))) * cell;
    let size = cell * (0.1 + 0.08 * roll);
    let reach = length(vec2<f32>(along - across, (up - lift) * 0.8));
    return (1.0 - smoothstep(size - grain, size + grain, reach)) * (1.0 - smoothstep(0.3, 0.7, grain / size));
}

fn sheet_crossed(item: EffectItem, world: vec3<f32>) -> bool {
    let axes = effect_axes(item);
    let half = vec3<f32>(item.p0.w, item.p1.y * 0.5, item.p1.x);
    let centre = item.p0.xyz + item.p5.xyz * half.y;
    let eye = (camera.position.xyz - centre) * axes;
    let toward = stable_normalize(world - camera.position.xyz) * axes;
    let inverse = 1.0 / select(toward, vec3<f32>(1.0e-9), abs(toward) < vec3<f32>(1.0e-9));
    let a = (-half - eye) * inverse;
    let b = (half - eye) * inverse;
    let near = max(max(min(a.x, b.x), min(a.y, b.y)), min(a.z, b.z));
    let far = min(min(max(a.x, b.x), max(a.y, b.y)), max(a.z, b.z));
    return near <= far && far > 0.0;
}

fn smoothed(share: f32) -> f32 {
    let clamped = saturate(share);
    return clamped * clamped * (3.0 - 2.0 * clamped);
}

@fragment fn fs_sheet(corner: SheetCorner) -> @location(0) vec4<f32> {
    let item = effect_items[corner.index];
    let along = corner.local.x;
    let up = corner.local.y;
    let height = item.p1.y;
    let seed = item.p2.x;
    let peel = effects_frame.curtain_peel;
    let pulp = effects_frame.curtain_pulp;
    let foam = effects_frame.curtain_foam;
    let shape = effects_frame.curtain_shape;
    let streak_m = max(peel.w, 0.02);
    let fine_m = streak_m * 0.3;
    let splash_m = max(shape.y, 0.02);
    let flow = up + effect_seconds() * shape.x;
    let grain = max(length(fwidth(corner.world)), 1.0e-4);
    let coarse = sheet_streaks(along, flow, streak_m, seed);
    let fine = mix(sheet_streaks(along, flow * 1.4, fine_m, seed + 9.0), 0.5,
        smoothstep(0.15, 0.4, grain / fine_m));
    let stream = effect_noise(vec3<f32>(along / (streak_m * 0.8), flow / (streak_m * 10.0), seed + 3.0));
    let drip = effect_noise(vec3<f32>(along / (streak_m * 0.6), seed, 1.0));
    let churn = effect_noise(vec3<f32>(along / (splash_m * 0.6), effect_seconds() * 3.0, seed + 23.0));
    let big = sheet_fleck(vec2<f32>(along, flow * 1.1), max(foam.w, 0.01), seed + 11.0, grain);
    let small = sheet_fleck(vec2<f32>(along, flow * 1.25), max(foam.w * 0.4, 0.005), seed + 17.0, grain);
    let drops = sheet_drops(along, up, seed + 5.0, grain);

    let past_end = abs(along) - item.p0.w;
    let end_share = select(1.0, 1.0 - past_end / max(item.p1.z * (0.35 + 0.65 * stream), 0.01), past_end > 0.0);
    let past_top = up - height;
    let top_share = select(1.0, 1.0 - past_top / max(item.p1.w * (0.3 + 0.7 * drip), 0.01), past_top > 0.0);
    let total = height + item.p1.w;
    let wander = (stream - 0.5) * streak_m * 2.0;
    let front = mix(total, -streak_m * 2.0, item.p2.y) + wander;
    let lip = mix(total + streak_m * 2.0, -streak_m * 2.0, item.p2.z) + wander;
    let flowing = smoothstep(front - grain, front + streak_m, up) * (1.0 - smoothstep(lip - streak_m, lip + grain, up));
    let crossed = select(0.0, 1.0, sheet_crossed(item, corner.world));
    let alpha = max(smoothed(end_share) * smoothed(top_share), crossed) * flowing;
    if alpha < 0.01 {
        discard;
    }

    let thin = 1.0 - alpha;
    let body = mix(pulp.rgb, peel.rgb, 0.55 + 0.45 * coarse);
    var color = mix(body, pulp.rgb * 1.15, thin * 0.7) * (0.8 + 0.4 * fine);
    color = mix(color, mix(pulp.rgb, foam.rgb, 0.5), max(big, small * 0.8));
    let foamy = 1.0 - smoothstep(splash_m * 0.35, splash_m, up + (churn - 0.5) * splash_m * 0.8);
    let line = exp(-max(up, 0.0) / (splash_m * 0.2));
    color = mix(color, foam.rgb, saturate(foamy * 0.8 + line * 0.5 + drops));

    let normal = stable_normalize(corner.normal);
    let view = stable_normalize(camera.position.xyz - corner.world);
    let sun = effects_frame.sun;
    let lit = sun.w != 0.0;
    var shown = effect_lit(color, normal, corner.field);
    let through = select(0.0, pow(saturate(dot(-view, sun.xyz)), 3.0), lit) * corner.field.a;
    shown += pulp.rgb * effects_frame.sun_color.rgb * through * (0.5 + thin);
    let tilt = stable_normalize(normal + item.p4.xyz * (fine - 0.5) * 0.6);
    let glint = select(0.0, pow(saturate(dot(tilt, stable_normalize(sun.xyz + view))), 48.0), lit);
    shown += effects_frame.sun_color.rgb * glint * corner.field.a * 0.5;
    shown += pulp.rgb * pulp.w;
    let rim = select(0.0, 4.0 * top_share * (1.0 - top_share), past_top > 0.0);
    shown += item.p3.rgb * rim * effects_frame.panel.w * 0.25;

    let near = 1.0 - smoothstep(0.2, max(shape.z, 0.25), distance(camera.position.xyz, corner.world));
    let close = effect_lit(body * (0.55 + 0.5 * fine), normal, corner.field) * 0.8;
    shown = mix(shown, close, near);
    return vec4<f32>(shown * alpha, alpha);
}
