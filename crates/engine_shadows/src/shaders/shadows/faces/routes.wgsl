import package::generated::shadows::{CasterCull, MAX_MOVING_TILES, SKY_REFRESH};
import package::shadows::sample::shadow_tiles;

struct Routes {
    clips: array<mat4x4<f32>, MAX_MOVING_TILES>,
    slots: array<vec4<f32>, MAX_MOVING_TILES>,
    atlas: vec4<f32>,
}

@group(0) @binding(0) var<uniform> caster_cull: CasterCull;
@group(0) @binding(0) var<storage, read> caster_routes: Routes;
@group(0) @binding(0) var<storage, read_write> caster_marks: array<atomic<u32>>;
@group(0) @binding(0) var<storage, read_write> caster_masks: array<atomic<u32>>;

fn overlaps(rectangle: vec4<f32>, low: vec3<f32>, high: vec3<f32>, depth: f32) -> bool {
    return all(high.xy >= rectangle.xy) && all(low.xy <= rectangle.zw) && low.z <= depth;
}

fn caster_box(center: vec3<f32>, half: vec3<f32>) -> array<vec3<f32>, 2> {
    let light_center = vec3<f32>(dot(caster_cull.light[0].xyz, center),
        dot(caster_cull.light[1].xyz, center), dot(caster_cull.light[2].xyz, center));
    let light_half = vec3<f32>(dot(abs(caster_cull.light[0].xyz), half),
        dot(abs(caster_cull.light[1].xyz), half), dot(abs(caster_cull.light[2].xyz), half));
    return array<vec3<f32>, 2>(light_center - light_half, light_center + light_half);
}

fn caster_regions() -> u32 {
    return caster_cull.counts.x;
}

fn caster_in_region(region: u32, low: vec3<f32>, high: vec3<f32>) -> bool {
    return overlaps(caster_cull.regions[region], low, high, caster_cull.region_depths[region].x);
}

fn caster_in_sky(center: vec3<f32>, half: vec3<f32>) -> bool {
    if (caster_cull.mode.x & SKY_REFRESH) == 0u {
        return false;
    }
    let sky_center = vec3<f32>(dot(caster_cull.sky[0].xyz, center),
        dot(caster_cull.sky[1].xyz, center), dot(caster_cull.sky[2].xyz, center));
    let sky_half = vec3<f32>(dot(abs(caster_cull.sky[0].xyz), half),
        dot(abs(caster_cull.sky[1].xyz), half), dot(abs(caster_cull.sky[2].xyz), half));
    return overlaps(caster_cull.sky_rect, sky_center - sky_half, sky_center + sky_half,
        caster_cull.sky_depths.x);
}

fn caster_levels() -> u32 {
    return caster_cull.counts.y;
}

fn caster_span(level: u32, low: vec2<f32>, high: vec2<f32>) -> vec4<i32> {
    if !(all(abs(low) < vec2<f32>(1.0e9)) && all(abs(high) < vec2<f32>(1.0e9))) {
        return vec4<i32>(0, 0, -1, -1);
    }
    let last = i32(caster_cull.counts.w) - 1;
    let window = caster_cull.windows[level];
    let first = vec2<i32>(round(window.xy / window.z));
    let from_tile = max(vec2<i32>(floor(low / window.z)), first);
    let to_tile = min(vec2<i32>(floor(high / window.z)), first + vec2<i32>(last));
    return vec4<i32>(from_tile, to_tile);
}

fn mark_caster(level: u32, tile: vec2<i32>, moving: bool) -> u32 {
    let side = caster_cull.counts.w;
    let last = i32(side) - 1;
    let cell = vec2<u32>(tile & vec2<i32>(last));
    let entry = textureLoad(shadow_tiles, vec2<u32>(cell.x, level * side + cell.y), 0).r;
    if (entry & 0xffffu) == 0u {
        return 0u;
    }
    let bit = (level * side + cell.y) * side + cell.x;
    let word = bit / 32u;
    let flag = 1u << (bit % 32u);
    if moving {
        if (atomicLoad(&caster_masks[word]) & flag) == 0u {
            atomicOr(&caster_masks[word], flag);
        }
    } else if (atomicLoad(&caster_marks[word]) & flag) == 0u {
        atomicOr(&caster_marks[word], flag);
    }
    let routed = entry >> 16u;
    if !moving || routed == 0u {
        return 0u;
    }
    return routed - caster_cull.moving.x;
}

fn route(world: vec3<f32>, tile: u32) -> vec4<f32> {
    let clip = caster_routes.clips[tile] * vec4<f32>(world, 1.0);
    return vec4<f32>(clip.xy, max(clip.z, 0.0), clip.w);
}

fn outside_route(position: vec2<f32>, tile: u32) -> bool {
    let slot = caster_routes.slots[tile];
    return any(position < slot.xy) || any(position >= slot.zw);
}
