import package::engine::view::{camera, stable_normalize, view_clip};
import package::effects::items::{EFFECT_HIDDEN, EFFECT_TAU, EffectItem, effect_corner,
    effect_direction, effect_items, effect_random};

const GLOW_STREAK: f32 = 0.0;
const GLOW_STAR: f32 = 1.0;
const GLOW_ROUND: f32 = 2.0;

struct GlowCorner {
    @builtin(position) clip: vec4<f32>,
    @location(0) unit: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) shape: vec4<f32>,
}

fn hidden() -> GlowCorner {
    var out: GlowCorner;
    out.clip = EFFECT_HIDDEN;
    out.unit = vec2<f32>(0.0);
    out.color = vec4<f32>(0.0);
    out.shape = vec4<f32>(0.0);
    return out;
}

fn facing_side(axis: vec3<f32>, point: vec3<f32>) -> vec3<f32> {
    let toward = camera.position.xyz - point;
    return stable_normalize(cross(axis, toward));
}

fn streak(back: vec3<f32>, front: vec3<f32>, half_width: f32, unit: vec2<f32>) -> vec3<f32> {
    let along = unit.x * 0.5 + 0.5;
    let point = mix(back, front, along);
    return point + facing_side(front - back, point) * half_width * unit.y;
}

fn billboard(centre: vec3<f32>, size: f32, roll: f32, unit: vec2<f32>) -> vec3<f32> {
    let toward = stable_normalize(camera.position.xyz - centre);
    let right = stable_normalize(cross(vec3<f32>(0.0, 1.0, 0.0), toward));
    let up = cross(toward, right);
    let turned = vec2<f32>(unit.x * cos(roll) - unit.y * sin(roll), unit.x * sin(roll) + unit.y * cos(roll));
    return centre + (right * turned.x + up * turned.y) * size;
}

fn lit(color: vec3<f32>, opacity: f32, fade: f32) -> vec4<f32> {
    let alpha = opacity * fade;
    return vec4<f32>(color * select(fade, alpha, opacity > 0.0), alpha);
}

fn tracer_corner(item: EffectItem, unit: vec2<f32>, fade: f32) -> GlowCorner {
    let halo = item.p5.x;
    let half_width = item.p2.w * (1.0 + halo);
    var out: GlowCorner;
    out.clip = view_clip(streak(item.p0.xyz, item.p1.xyz, half_width, unit));
    out.unit = unit;
    out.color = vec4<f32>(item.p2.rgb * fade, 0.0);
    out.shape = vec4<f32>(GLOW_STREAK, halo, 0.0, 1.0);
    return out;
}

fn flash_corner(item: EffectItem, unit: vec2<f32>, fade: f32) -> GlowCorner {
    let seed = u32(item.p3.y);
    let size = item.p2.w * (0.75 + 0.5 * effect_random(seed));
    var out: GlowCorner;
    out.clip = view_clip(billboard(item.p0.xyz, size, effect_random(seed + 7u) * EFFECT_TAU, unit));
    out.unit = unit;
    out.color = vec4<f32>(item.p2.rgb * fade * fade, 0.0);
    out.shape = vec4<f32>(GLOW_STAR, 0.0, item.p5.y, 0.0);
    return out;
}

fn halo_corner(item: EffectItem, unit: vec2<f32>) -> GlowCorner {
    let size = item.p2.w;
    let toward = stable_normalize(camera.position.xyz - item.p0.xyz);
    var out: GlowCorner;
    out.clip = view_clip(billboard(item.p0.xyz + toward * size, size, 0.0, unit));
    out.unit = unit;
    out.color = vec4<f32>(item.p2.rgb, 0.0);
    out.shape = vec4<f32>(GLOW_ROUND, 0.0, 0.0, 0.0);
    return out;
}

fn spray_corner(item: EffectItem, sprite: u32, unit: vec2<f32>, fade: f32) -> GlowCorner {
    let seed = u32(item.p3.y) + sprite * 7919u;
    let age = item.p0.w;
    let gravity = vec3<f32>(0.0, -item.p4.x, 0.0);
    let thrown = stable_normalize(item.p1.xyz + effect_direction(seed) * item.p4.z);
    let speed = item.p3.w * (0.4 + 0.6 * effect_random(seed + 2u));
    let start = thrown * speed;
    let place = item.p0.xyz + start * age + 0.5 * gravity * age * age;
    let moving = start + gravity * age;
    let size = item.p2.w * (0.6 + 0.8 * effect_random(seed + 3u)) * (0.5 + 0.5 * fade);
    var out: GlowCorner;
    out.unit = unit;
    out.color = lit(item.p2.rgb, item.p4.y, fade);
    let stretch = item.p4.w;
    if stretch > 0.0 {
        out.clip = view_clip(streak(place - moving * stretch, place, size, unit));
        out.shape = vec4<f32>(GLOW_STREAK, 0.0, 0.0, 0.0);
    } else {
        out.clip = view_clip(billboard(place, size, 0.0, unit));
        out.shape = vec4<f32>(GLOW_ROUND, 0.0, 0.0, 0.0);
    }
    return out;
}

@vertex fn vs_glow(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> GlowCorner {
    let item = effect_items[instance];
    let sprite = vertex / 6u;
    let age = item.p0.w;
    let life = item.p1.w;
    if sprite >= u32(item.p3.z) || age >= life || life <= 0.0 {
        return hidden();
    }
    let unit = effect_corner(vertex);
    let fade = 1.0 - age / life;
    let kind = u32(item.p3.x);
    if kind == 0u {
        return tracer_corner(item, unit, fade);
    }
    if kind == 1u {
        return flash_corner(item, unit, fade);
    }
    if kind == 3u {
        return halo_corner(item, unit);
    }
    return spray_corner(item, sprite, unit, fade);
}

@fragment fn fs_glow(corner: GlowCorner) -> @location(0) vec4<f32> {
    let unit = corner.unit;
    var strength = 0.0;
    if corner.shape.x == GLOW_STREAK {
        let across = abs(unit.y);
        let core = 1.0 - smoothstep(0.0, 1.0 / (1.0 + corner.shape.y), across);
        let soft = (1.0 - across) * (1.0 - across);
        let head = mix(1.0, mix(0.35, 1.0, unit.x * 0.5 + 0.5), corner.shape.w);
        strength = (core + soft * 0.35) * head;
    } else if corner.shape.x == GLOW_STAR {
        let reach = length(unit);
        let angle = atan2(unit.y, unit.x);
        let petals = pow(abs(cos(angle * corner.shape.z * 0.5)), 6.0);
        let edge = 0.35 + 0.65 * petals;
        strength = saturate(1.0 - reach / edge) + saturate(1.0 - reach) * 0.3;
    } else {
        let reach = dot(unit, unit);
        strength = saturate(1.0 - reach);
    }
    if strength <= 0.0 {
        discard;
    }
    return corner.color * strength;
}
