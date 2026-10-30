import package::face::shadows::KeptLift;
import package::frame::view::{DetailLevel, PI, detail_across, detail_level, fade_noise};

struct Shadow {
    light: array<vec4<f32>, 3>,
    state: vec4<f32>,
    level_state: vec4<f32>,
    atlas: vec4<f32>,
    windows: array<vec4<f32>, SHADOW_LEVELS>,
    bands: array<vec4<f32>, SHADOW_LEVELS>,
}
const SHADOW_LEVELS: u32 = 20u;
const SHADOW_APRON: u32 = 1u;
const KEPT_GROUND_TEXELS: f32 = 2.0;
@group(0) @binding(2) var<uniform> shadow: Shadow;
@group(0) @binding(3) var shadow_atlas: texture_depth_2d;
@group(0) @binding(4) var shadow_tiles: texture_2d<u32>;
@group(0) @binding(6) var shadow_sampler: sampler_comparison;
@group(0) @binding(7) var shadow_texels: sampler;

override SHADOW_MAPS: bool = true;

const RECEIVER_MARGIN: f32 = 2.0 / 65535.0;

struct MapPoint {
    texel: vec2<f32>,
    depth: f32,
    slope: vec2<f32>,
    soft: f32,
}

var<private> crown_light: f32;

fn crown_softness(per_meter: f32) -> f32 {
    return crown_light * shadow.state.y * abs(per_meter);
}

fn texel_slope(map_dx: vec4<f32>, map_dy: vec4<f32>, scale: vec2<f32>, limit: f32) -> vec2<f32> {
    let texel_dx = map_dx.xy * scale;
    let texel_dy = map_dy.xy * scale;
    let det = texel_dx.x * texel_dy.y - texel_dx.y * texel_dy.x;
    var slope = vec2<f32>(0.0);
    if abs(det) > 1.0e-12 {
        slope = vec2<f32>(map_dx.z * texel_dy.y - map_dy.z * texel_dx.y,
            map_dy.z * texel_dx.x - map_dx.z * texel_dy.x) / det;
    }
    return clamp(slope, vec2<f32>(-limit), vec2<f32>(limit));
}

fn atlas_depths(point: MapPoint) -> vec4<f32> {
    let corner = floor(point.texel - vec2<f32>(0.5)) + vec2<f32>(1.0);
    return textureGather(shadow_atlas, shadow_texels, corner * shadow.atlas.zw);
}

fn plane_visibility(depths: vec4<f32>, point: MapPoint) -> f32 {
    let corner = point.texel - vec2<f32>(0.5);
    let f = corner - floor(corner);
    let du = vec4<f32>(-f.x, 1.0 - f.x, 1.0 - f.x, -f.x);
    let dv = vec4<f32>(1.0 - f.y, 1.0 - f.y, -f.y, -f.y);
    let plane = point.depth + point.slope.x * du + point.slope.y * dv - RECEIVER_MARGIN;
    var lit = select(vec4<f32>(0.0), vec4<f32>(1.0), plane <= depths);
    if point.soft > 0.0 {
        lit = exp2(-max(plane - depths, vec4<f32>(0.0)) / point.soft);
    }
    return mix(mix(lit.w, lit.z, f.x), mix(lit.x, lit.y, f.x), f.y);
}

fn light_space(world: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(shadow.light[0].xyz, world), dot(shadow.light[1].xyz, world),
        dot(shadow.light[2].xyz, world));
}

fn shadow_level(world: vec3<f32>) -> DetailLevel {
    return detail_level(detail_across(world), shadow.level_state.y, u32(shadow.level_state.x));
}

struct LevelPoint {
    cell: vec2<u32>,
    across: vec2<f32>,
    depth: f32,
    inside: bool,
}

fn level_point(level: u32, light: vec3<f32>) -> LevelPoint {
    let window = shadow.windows[level];
    let band = shadow.bands[level];
    let side = shadow.atlas.x;
    let tiles = (light.xy - window.xy) / window.z;
    let tile = floor(tiles);
    var point: LevelPoint;
    point.depth = (light.z - band.x) * band.y;
    point.inside = all(tile >= vec2<f32>(0.0)) && all(tile < vec2<f32>(side))
        && point.depth > 0.0 && point.depth < 1.0;
    let first = vec2<i32>(round(window.xy / window.z));
    point.cell = vec2<u32>((first + vec2<i32>(tile)) & vec2<i32>(i32(side) - 1));
    point.across = tiles - tile;
    return point;
}

fn level_slots(level: u32, point: LevelPoint) -> vec2<u32> {
    let side = u32(shadow.atlas.x);
    let entry = textureLoad(shadow_tiles, vec2<u32>(point.cell.x, level * side + point.cell.y),
        0).r;
    return vec2<u32>(entry & 0xffffu, entry >> 16u);
}

fn slot_texel(slot: u32, across: vec2<f32>) -> vec2<f32> {
    let texels = shadow.level_state.z;
    let side = texels + 2.0 * f32(SHADOW_APRON);
    let per_row = u32(shadow.atlas.y);
    let corner = vec2<f32>(vec2<u32>(slot % per_row, slot / per_row)) * side;
    return corner + vec2<f32>(f32(SHADOW_APRON)) + vec2<f32>(across.x, 1.0 - across.y) * texels;
}

var<private> kept_lift: KeptLift;

fn level_lift(level: u32) -> f32 {
    if level < kept_lift.first || level - kept_lift.first >= 4u {
        return 0.0;
    }
    return kept_lift.lifts[level - kept_lift.first];
}

const CROWN_TAPS: u32 = 4u;
const CROWN_REACH: f32 = 2.0;
const CROWN_SPIRAL_TURN: f32 = 2.39996323;

fn crown_spread(level: u32, light: vec3<f32>, point: MapPoint, center: LevelPoint, slot: u32,
        visible: f32) -> f32 {
    let depth_per_meter = max(shadow.bands[level].y, 1.0e-12);
    let crown = point.soft / depth_per_meter;
    let turn = fade_noise(floor(point.texel) + vec2<f32>(0.5)) * 2.0 * PI;
    var way = vec2<f32>(cos(turn), sin(turn));
    let step = vec2<f32>(cos(CROWN_SPIRAL_TURN), sin(CROWN_SPIRAL_TURN));
    var lit = visible;
    for (var tap = 0u; tap < CROWN_TAPS; tap++) {
        let across = CROWN_REACH * crown * sqrt((f32(tap) + 0.5) / f32(CROWN_TAPS));
        let at = level_point(level, light + vec3<f32>(way, 0.0) * across);
        way = vec2<f32>(way.x * step.x - way.y * step.y, way.x * step.y + way.y * step.x);
        if !at.inside {
            continue;
        }
        var held = slot;
        if any(at.cell != center.cell) {
            held = level_slots(level, at).x;
            if held == 0u {
                continue;
            }
        }
        let caster = textureLoad(shadow_atlas, vec2<i32>(slot_texel(held - 1u, at.across)), 0);
        let down = max(point.depth - caster, 0.0) / depth_per_meter;
        lit = max(lit, exp2(-(across + down) / crown));
    }
    return lit;
}

fn level_visibility(level: u32, world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>,
        dy: vec3<f32>, spread: bool) -> f32 {
    let window = shadow.windows[level];
    let band = shadow.bands[level];
    let up = vec3<f32>(shadow.light[0].y, shadow.light[1].y, shadow.light[2].y);
    let light = light_space(world + normal * band.z) + up * level_lift(level);
    let at = level_point(level, light);
    if !at.inside {
        return -1.0;
    }
    let slots = level_slots(level, at);
    if slots.x == 0u {
        return -1.0;
    }
    var point: MapPoint;
    point.texel = slot_texel(slots.x - 1u, at.across);
    point.depth = at.depth;
    let per_world = vec3<f32>(1.0, 1.0, band.y);
    point.slope = texel_slope(vec4<f32>(light_space(dx) * per_world, 0.0),
        vec4<f32>(light_space(dy) * per_world, 0.0), vec2<f32>(1.0, -1.0) / window.w, band.w);
    point.soft = crown_softness(band.y);
    var visible = plane_visibility(atlas_depths(point), point);
    if spread && point.soft > 0.0 && visible < 1.0 {
        visible = crown_spread(level, light, point, at, slots.x, visible);
    }
    if slots.y != 0u {
        var moving = point;
        moving.texel = slot_texel(slots.y - 1u, at.across);
        visible = min(visible, plane_visibility(atlas_depths(moving), moving));
    }
    return visible;
}

fn map_lookup(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>) -> vec2<f32> {
    let pick = shadow_level(world);
    let count = u32(shadow.level_state.x);
    for (var level = pick.level; level < count; level++) {
        let visible = level_visibility(level, world, normal, dx, dy, true);
        if visible < 0.0 {
            continue;
        }
        if level > pick.level || pick.blend <= 0.0 {
            return vec2<f32>(visible, f32(level));
        }
        for (var next = level + 1u; next < count; next++) {
            let coarser = level_visibility(next, world, normal, dx, dy, false);
            if coarser >= 0.0 {
                return vec2<f32>(mix(visible, coarser, pick.blend), f32(level));
            }
        }
        return vec2<f32>(visible, f32(level));
    }
    return vec2<f32>(1.0, -1.0);
}

fn map_visibility(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>) -> f32 {
    return map_lookup(world, normal, dx, dy).x;
}

fn air_visibility(world: vec3<f32>) -> f32 {
    let pick = shadow_level(world);
    let light = light_space(world);
    for (var level = pick.level; level < u32(shadow.level_state.x); level++) {
        let at = level_point(level, light);
        if !at.inside {
            continue;
        }
        let slots = level_slots(level, at);
        if slots.x == 0u {
            continue;
        }
        let still = slot_texel(slots.x - 1u, at.across) * shadow.atlas.zw;
        var visible = textureSampleCompareLevel(shadow_atlas, shadow_sampler, still, at.depth);
        if slots.y != 0u {
            let moving = slot_texel(slots.y - 1u, at.across) * shadow.atlas.zw;
            visible = min(visible,
                textureSampleCompareLevel(shadow_atlas, shadow_sampler, moving, at.depth));
        }
        return visible;
    }
    return 1.0;
}
