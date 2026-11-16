import package::cover::faces::cover::COVER_UNBOUND;
import package::face::ground_tiles::{GroundTerrain, LevelTile, NOT_RESIDENT, TILE_QUADS,
    ground_extent, level_tile, tile_gradient, tile_height, tile_site, tile_summary};
import package::generated::cover::Grow;

const TILE_SPAN: f32 = f32(TILE_QUADS);
const TILE_LAST: u32 = TILE_QUADS - 1u;

struct Ground {
    height: f32,
    found: bool,
}

fn height_on(tile: LevelTile, samples: vec2<f32>) -> f32 {
    let corner = min(vec2<u32>(samples), tile.first + vec2<u32>(255u));
    let f = saturate(samples - vec2<f32>(corner));
    let a = tile_height(tile, corner);
    let b = tile_height(tile, corner + vec2<u32>(1u, 0u));
    let c = tile_height(tile, corner + vec2<u32>(0u, 1u));
    let d = tile_height(tile, corner + vec2<u32>(1u, 1u));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn ground_at(terrain: GroundTerrain, local: vec2<f32>) -> Ground {
    var ground: Ground;
    if any(local > ground_extent(terrain)) {
        return ground;
    }
    for (var index = 0u; index < terrain.levels; index++) {
        let samples = max(local / (terrain.spacing * f32(1u << index)), vec2<f32>(0.0));
        let tile = level_tile(terrain, index, samples);
        if tile.layer == NOT_RESIDENT {
            continue;
        }
        ground.height = height_on(tile, samples);
        ground.found = true;
        return ground;
    }
    return ground;
}

fn ordered_height(height: f32) -> u32 {
    let bits = bitcast<u32>(height);
    return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn ordered_to_height(ordered: u32) -> f32 {
    return bitcast<f32>(select(~ordered, ordered & 0x7fffffffu, (ordered & 0x80000000u) != 0u));
}

struct CellSamples {
    first: vec2<u32>,
    count: vec2<u32>,
}

fn cell_samples(grow: Grow, corner: vec2<f32>, spacing: f32, level: u32, first: vec2<u32>,
        reach: f32) -> CellSamples {
    let per = grow.origin.w * f32(1u << level);
    let low = max(vec2<u32>(max(floor((corner - vec2<f32>(reach)) / per - 0.001),
        vec2<f32>(0.0))), first);
    let high = vec2<u32>(floor((corner + vec2<f32>(spacing + reach)) / per + 0.001)) + 1u;
    var rect: CellSamples;
    rect.first = low;
    rect.count = high + 1u - min(low, high + 1u);
    return rect;
}

fn ground_normal(grow: Grow, terrain: GroundTerrain, local: vec2<f32>) -> vec3<f32> {
    for (var index = 0u; index < grow.counts.y; index++) {
        let per = grow.origin.w * f32(1u << index);
        let samples = max(local / per, vec2<f32>(0.0));
        let tile = level_tile(terrain, index, samples);
        if tile.layer == NOT_RESIDENT {
            continue;
        }
        let corner = min(vec2<u32>(samples), tile.first + vec2<u32>(255u));
        let f = saturate(samples - vec2<f32>(corner));
        let a = tile_gradient(tile, corner);
        let b = tile_gradient(tile, corner + vec2<u32>(1u, 0u));
        let c = tile_gradient(tile, corner + vec2<u32>(0u, 1u));
        let d = tile_gradient(tile, corner + vec2<u32>(1u, 1u));
        let gradient = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        return normalize(vec3<f32>(-gradient.x, 1.0, -gradient.y));
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

struct Between {
    corner: vec2<u32>,
    f: vec2<f32>,
}

fn tile_between(tile: LevelTile, samples: vec2<f32>) -> Between {
    let low = vec2<f32>(tile.first);
    let within = clamp(samples, low, low + vec2<f32>(TILE_SPAN));
    let corner = min(vec2<u32>(within), tile.first + vec2<u32>(TILE_LAST));
    return Between(corner, within - vec2<f32>(corner));
}

fn site_between(tile: LevelTile, at: Between) -> vec4<f32> {
    let low = mix(tile_site(tile, at.corner), tile_site(tile, at.corner + vec2<u32>(1u, 0u)),
        at.f.x);
    let high = mix(tile_site(tile, at.corner + vec2<u32>(0u, 1u)),
        tile_site(tile, at.corner + vec2<u32>(1u, 1u)), at.f.x);
    return mix(low, high, at.f.y);
}

fn site_lush(low: vec4<f32>, high: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(saturate(0.5 + 0.25 * (low.x + low.z - high.y - high.w)),
        saturate(0.5 + 0.25 * (high.x + high.z - low.y - low.w)));
}

struct Under {
    made: bool,
    heights: vec2<f32>,
    named: u32,
    low: vec4<f32>,
    high: vec4<f32>,
    wide_low: vec4<f32>,
    wide_high: vec4<f32>,
}

fn summed_under(tile: LevelTile, start: vec2<i32>, wide: u32, extra: u32) -> Under {
    let count = i32(max(wide, 1u));
    let reach = i32(extra);
    let mip = 32u - countLeadingZeros(u32(count + 2 * reach - 3));
    let edge = vec2<i32>(i32(TILE_QUADS) - 1);
    let first = vec2<u32>(clamp(start + vec2<i32>(1 - reach), vec2<i32>(0), edge))
        >> vec2<u32>(mip);
    let last = vec2<u32>(clamp(start + vec2<i32>(count + reach - 2), vec2<i32>(0), edge))
        >> vec2<u32>(mip);
    var under = Under(true, vec2<f32>(COVER_UNBOUND, -COVER_UNBOUND), 0u, vec4<f32>(1.0),
        vec4<f32>(0.0), vec4<f32>(1.0), vec4<f32>(0.0));
    for (var y = first.y; y <= last.y; y++) {
        for (var x = first.x; x <= last.x; x++) {
            let summary = tile_summary(tile, vec2<u32>(x, y), mip);
            under.made = under.made && summary.materials != 0u;
            under.heights = vec2<f32>(min(under.heights.x, summary.wide_lowest),
                max(under.heights.y, summary.wide_highest));
            under.named |= summary.wide_materials;
            under.low = min(under.low, summary.site_low);
            under.high = max(under.high, summary.site_high);
            under.wide_low = min(under.wide_low, summary.wide_site_low);
            under.wide_high = max(under.wide_high, summary.wide_site_high);
        }
    }
    return under;
}
