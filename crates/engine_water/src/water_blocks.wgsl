import package::place::heightfield::{WaterCell, WaterLevel};
import package::terrain::faces::heightfield::{NO_WATER_BLOCK, TERRAIN_GRID, WATER_SPAN_LEAST,
    WATER_SPAN_MOST, WATER_TILE_PATCHES, terrain, water_smooth, water_texel};
import package::terrain::field::terrain_directory;

@group(1) @binding(50) var water_surface: texture_2d_array<f32>;

const WATER_BLOCK: u32 = 19u;
const WATER_BLOCKS_SIDE: u32 = 16u;
const WATER_TILE_WORDS: u32 = 256u;
const WATER_FLOW_RANGE: f32 = 8.0;
const WATER_SHEAR_RANGE: f32 = 8.0;

fn water_span(code: f32) -> f32 {
    let steps = round(abs(code) * 127.0);
    let octaves = log2(WATER_SPAN_MOST / WATER_SPAN_LEAST);
    return select(WATER_SPAN_LEAST * exp2(octaves * (steps - 1.0) / 126.0), 0.0, steps < 0.5);
}

fn water_extent() -> vec2<f32> {
    let header = terrain.directory.x;
    let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
    return vec2<f32>(tiles * (WATER_TILE_PATCHES * (TERRAIN_GRID - 1u))) * terrain.anchor.w;
}

struct WaterSpline {
    weights: vec4<f32>,
    rates: vec4<f32>,
}

fn water_spline(t: f32) -> WaterSpline {
    let s = 1.0 - t;
    var spline: WaterSpline;
    spline.weights = vec4<f32>(s * s * s, (3.0 * t - 6.0) * t * t + 4.0,
        ((3.0 - 3.0 * t) * t + 3.0) * t + 1.0, t * t * t) / 6.0;
    spline.rates = vec4<f32>(-s * s, (3.0 * t - 4.0) * t, (2.0 - 3.0 * t) * t + 1.0, t * t)
        * 0.5;
    return spline;
}

fn water_level(cell: WaterCell) -> WaterLevel {
    if cell.block == NO_WATER_BLOCK {
        var level: WaterLevel;
        return level;
    }
    let origin = water_texel(cell.block, cell.first - vec2<u32>(1u));
    var held: array<vec2<f32>, 16>;
    for (var k = 0u; k < 16u; k++) {
        let steps = vec2<i32>(vec2<u32>(k % 4u, k / 4u));
        held[k] = textureLoad(water_surface, origin.xy + steps, origin.z, 0).xy;
    }
    return water_smooth(held, cell.within, terrain.anchor.w * f32(1u << cell.level));
}
