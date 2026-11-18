import package::engine::view::{camera, stable_normalize, view_clip};
import package::effects::items::{EFFECT_TAU, EffectItem, effect_axes, effect_box_corner,
    effect_items, effect_lit, effect_noise, effect_seconds};

@group(0) @binding(1) var effects_depth: texture_depth_2d;

const VOLUME_FAR: f32 = 1.0e9;

struct VolumeCorner {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) @interpolate(flat) index: u32,
}

struct Seen {
    eye: vec3<f32>,
    toward: vec3<f32>,
    own: f32,
    scene: f32,
}

@vertex fn vs_volume(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> VolumeCorner {
    let item = effect_items[instance];
    let corner = effect_box_corner(vertex);
    let world = item.p0.xyz + effect_axes(item) * (corner.local * item.p1.xyz);
    var out: VolumeCorner;
    out.clip = view_clip(world);
    out.world = world;
    out.index = instance;
    return out;
}

fn scene_distance(pixel: vec2<f32>, eye: vec3<f32>, world: vec3<f32>) -> f32 {
    let depth = textureLoad(effects_depth, vec2<i32>(pixel), 0);
    let toward = world - eye;
    let start = camera.view_projection * vec4<f32>(eye, 1.0);
    let step = camera.view_projection * vec4<f32>(toward, 0.0);
    let below = step.z - depth * step.w;
    if depth <= 0.0 || below == 0.0 {
        return VOLUME_FAR;
    }
    let t = (depth * start.w - start.z) / below;
    return select(VOLUME_FAR, t * length(toward), t > 0.0);
}

fn far_face(item: EffectItem, seen: Seen) -> bool {
    let axes = effect_axes(item);
    let local_eye = (seen.eye - item.p0.xyz) * axes;
    let local_toward = seen.toward * axes;
    let half = item.p1.xyz;
    let inverse = 1.0 / select(local_toward, vec3<f32>(1.0e-9), abs(local_toward) < vec3<f32>(1.0e-9));
    let a = (-half - local_eye) * inverse;
    let b = (half - local_eye) * inverse;
    let near = max(max(min(a.x, b.x), min(a.y, b.y)), min(a.z, b.z));
    let far = min(min(max(a.x, b.x), max(a.y, b.y)), max(a.z, b.z));
    return seen.own >= (near + far) * 0.5;
}

fn seen_from(corner: VolumeCorner, pixel: vec2<f32>) -> Seen {
    let eye = camera.position.xyz;
    let offset = corner.world - eye;
    return Seen(eye, stable_normalize(offset), length(offset), scene_distance(pixel, eye, corner.world));
}

@fragment fn fs_smoke(corner: VolumeCorner) -> @location(0) vec4<f32> {
    let item = effect_items[corner.index];
    let seen = seen_from(corner, corner.clip.xy);
    if !far_face(item, seen) {
        discard;
    }
    let centre = item.p0.xyz;
    let radius = item.p0.w;
    let rim = item.p2.w;
    let offset = centre - seen.eye;
    let closest = dot(offset, seen.toward);
    let miss_squared = max(dot(offset, offset) - closest * closest, 0.0);
    let swirl = seen.eye + seen.toward * closest;
    let drift = vec3<f32>(0.0, effect_seconds() * item.p5.w, item.p3.y * 17.0);
    let noise = effect_noise(swirl * item.p4.w + drift);
    let reach = radius + rim * (0.5 + 0.5 * noise);
    let half_squared = reach * reach - miss_squared;
    if half_squared <= 0.0 {
        discard;
    }
    let half_chord = sqrt(half_squared);
    let entry = max(closest - half_chord, 0.0);
    let leave = min(closest + half_chord, seen.scene);
    let through = leave - entry;
    if through <= 0.0 {
        discard;
    }
    let density = 3.0 / max(item.p3.z, 1.0e-3) * max(item.p3.w, 0.05);
    let alpha = 1.0 - exp(-density * through);
    let surface = seen.eye + seen.toward * entry - centre;
    let normal = select(-seen.toward, surface / max(length(surface), 1.0e-3), closest - half_chord > 0.0);
    let shade = 0.85 + 0.3 * noise;
    let color = effect_lit(item.p2.rgb * shade, normal);
    return vec4<f32>(color * alpha, alpha);
}

fn wall_glow(item: EffectItem, seen: Seen, radius: f32) -> f32 {
    let axes = effect_axes(item);
    let eye = (seen.eye - item.p0.xyz) * axes;
    let toward = seen.toward * axes;
    let a = dot(toward.xz, toward.xz);
    let b = dot(eye.xz, toward.xz);
    let c = dot(eye.xz, eye.xz) - radius * radius;
    let reach = b * b - a * c;
    if a <= 1.0e-6 || reach <= 0.0 {
        return 0.0;
    }
    let root = sqrt(reach);
    let height = item.p1.y;
    var most = 0.0;
    for (var side = 0u; side < 2u; side += 1u) {
        let t = (-b + select(-root, root, side == 1u)) / a;
        let y = eye.y + t * toward.y;
        if t > 0.0 && t < seen.scene && y >= 0.0 && y <= height {
            let up = 1.0 - y / height;
            most = max(most, up * up);
        }
    }
    return most;
}

fn ring(distance: f32, at: f32, width: f32) -> f32 {
    return 1.0 - smoothstep(0.0, max(width, 1.0e-3), abs(distance - at));
}

fn marked(item: EffectItem, local: vec3<f32>, seen: Seen) -> f32 {
    let kind = u32(item.p3.x);
    let radius = item.p0.w;
    let rim = item.p3.w;
    let fill = item.p2.w;
    let progress = item.p6.x;
    if kind == 4u {
        return smoothstep(radius, radius + rim, seen.scene) * fill;
    }
    if seen.scene >= VOLUME_FAR || any(abs(local) > item.p1.xyz) {
        return 0.0;
    }
    if kind == 5u {
        let spread = length(local.xz) / radius;
        let soft = 1.0 - smoothstep(0.25, 1.0, spread);
        let below = 1.0 - abs(local.y) / item.p1.y;
        return fill * soft * soft * below;
    }
    if kind == 0u {
        let spot = length(local.xy) / radius;
        let ragged = 0.65 + 0.35 * effect_noise(vec3<f32>(local.xy / radius * 3.0, item.p3.y));
        if spot > ragged {
            return 0.0;
        }
        return fill * item.p6.y * (1.0 - smoothstep(ragged * 0.5, ragged, spot));
    }
    let across = length(local.xz);
    if across > radius {
        return 0.0;
    }
    if kind == 1u {
        let wave = 0.5 + 0.5 * sin((across - effect_seconds() * item.p5.w) / max(item.p4.w, 1.0e-3) * EFFECT_TAU);
        let edge = smoothstep(radius - rim, radius, across);
        return max(fill * (0.6 + 0.4 * wave), edge);
    }
    if kind == 2u {
        let closing = ring(across, radius * (1.0 - progress), rim);
        let edge = ring(across, radius - rim * 0.5, rim);
        return max(max(edge, closing), fill * (0.5 + 0.5 * progress));
    }
    let front = radius * progress;
    let inside = select(0.0, fill, across <= front);
    return max(ring(across, front, rim), inside) * (1.0 - progress);
}

@fragment fn fs_marks(corner: VolumeCorner) -> @location(0) vec4<f32> {
    let item = effect_items[corner.index];
    let seen = seen_from(corner, corner.clip.xy);
    if !far_face(item, seen) {
        discard;
    }
    let point = seen.eye + seen.toward * seen.scene;
    let local = (point - item.p0.xyz) * effect_axes(item);
    let kind = u32(item.p3.x);
    var wall = 0.0;
    if kind == 1u || kind == 2u {
        wall = wall_glow(item, seen, item.p0.w) * 0.85;
    } else if kind == 3u {
        wall = wall_glow(item, seen, max(item.p0.w * item.p6.x, 1.0e-3)) * (1.0 - item.p6.x);
    }
    let alpha = saturate(max(marked(item, local, seen), wall));
    if alpha <= 0.0 {
        discard;
    }
    return vec4<f32>(item.p2.rgb * alpha, alpha);
}
