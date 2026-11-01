import package::face::ground_tiles::{GroundTerrain, NOT_RESIDENT, level_tiles, tile_layer};
import package::face::heightfield::{TERRAIN_GRID, terrain_uniform};
import package::generated::water::{WATER_BLOCK, WATER_BLOCKS_SIDE, WATER_FLOW_RANGE,
    WATER_SHEAR_RANGE, WATER_TILE_WORDS};
import package::place::water::{NO_WATER_BLOCK, NO_WATER_SURFACE, WATER_SPAN_LEAST, WATER_SPAN_MOST,
    WATER_TILE_PATCHES, WaterAt, WaterCell, WaterLevel, WaterWords};

@group(0) @binding(0) var water_surface: texture_2d_array<f32>;

@group(0) @binding(0) var water_info: texture_2d_array<u32>;

@group(0) @binding(0) var<storage, read> water_directory: array<u32>;

struct WaterSpline {
    weights: vec4<f32>,
    rates: vec4<f32>,
}

fn water_ground() -> GroundTerrain {
    let terrain = terrain_uniform();
    return GroundTerrain(terrain.directory.x, terrain.directory.y, terrain.anchor.w);
}

fn water_span(code: f32) -> f32 {
    let steps = round(abs(code) * 127.0);
    let octaves = log2(WATER_SPAN_MOST / WATER_SPAN_LEAST);
    return select(WATER_SPAN_LEAST * exp2(octaves * (steps - 1.0) / 126.0), 0.0, steps < 0.5);
}

fn water_extent() -> vec2<f32> {
    let tiles = level_tiles(water_ground(), 0u);
    return vec2<f32>(tiles * (WATER_TILE_PATCHES * (TERRAIN_GRID - 1u)))
        * terrain_uniform().anchor.w;
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
    return water_smooth(held, cell.within, terrain_uniform().anchor.w * f32(1u << cell.level));
}

fn water_info_at(texel: vec3<i32>) -> vec4<u32> {
    return textureLoad(water_info, texel.xy, texel.z, 0);
}

fn water_texel(block: u32, sample: vec2<u32>) -> vec3<i32> {
    let blocks = WATER_BLOCKS_SIDE * WATER_BLOCKS_SIDE;
    let index = block % blocks;
    let origin = vec2<u32>(index % WATER_BLOCKS_SIDE, index / WATER_BLOCKS_SIDE) * WATER_BLOCK;
    return vec3<i32>(vec2<i32>(origin + sample), i32(block / blocks));
}

fn water_words(words: vec4<u32>) -> WaterWords {
    let info = unpack4x8snorm(words.x);
    let state = unpack4x8unorm(words.y);
    var read: WaterWords;
    read.flow = info.xy * WATER_FLOW_RANGE;
    read.span = water_span(info.w);
    read.lake = info.w < 0.0;
    read.sheet = info.z < 0.0;
    read.state = vec3<f32>(state.xy, state.z * WATER_SHEAR_RANGE);
    read.reach = words.y >> 24u;
    return read;
}

fn water_cell(local: vec2<f32>, level: u32) -> WaterCell {
    let ground = water_ground();
    let levels = ground.levels;
    var cell: WaterCell;
    cell.level = levels;
    cell.block = NO_WATER_BLOCK;
    if any(local < vec2<f32>(0.0)) || any(local > water_extent()) {
        return cell;
    }
    let quads = TERRAIN_GRID - 1u;
    let tile_quads = WATER_TILE_PATCHES * quads;
    for (var index = level; index < levels; index++) {
        let samples = local / (ground.spacing * f32(1u << index));
        let tiles = level_tiles(ground, index);
        let tile = min(vec2<u32>(samples / f32(tile_quads)), tiles - vec2<u32>(1u));
        let layer = tile_layer(ground, index, tile);
        if layer == NOT_RESIDENT {
            continue;
        }
        let within = clamp(samples - vec2<f32>(tile * tile_quads), vec2<f32>(0.0),
            vec2<f32>(f32(tile_quads)));
        let corner = min(vec2<u32>(within), vec2<u32>(tile_quads - 1u));
        let tile_patch = corner / quads;
        cell.level = index;
        cell.block = water_directory[layer * WATER_TILE_WORDS
            + tile_patch.y * WATER_TILE_PATCHES + tile_patch.x];
        cell.first = corner - tile_patch * quads + vec2<u32>(1u);
        cell.within = within - vec2<f32>(corner);
        return cell;
    }
    return cell;
}

fn water_smooth(surfaces: array<vec2<f32>, 16>, within: vec2<f32>, spacing: f32)
        -> WaterLevel {
    var samples = surfaces;
    var level: WaterLevel;
    let along = water_spline(within.x);
    let across = water_spline(within.y);
    var counted = 0.0;
    var counting = false;
    var total = 0.0;
    var raised = 0.0;
    var deep = 0.0;
    var rates = vec2<f32>(0.0);
    var rises = vec2<f32>(0.0);
    for (var row = 0u; row < 4u; row++) {
        for (var column = 0u; column < 4u; column++) {
            let held = samples[row * 4u + column];
            if held.x <= NO_WATER_SURFACE * 0.5 {
                continue;
            }
            if !counting {
                counted = held.x;
                counting = true;
            }
            let weight = along.weights[column] * across.weights[row];
            let rate = vec2<f32>(along.rates[column] * across.weights[row],
                along.weights[column] * across.rates[row]);
            total += weight;
            raised += (held.x - counted) * weight;
            deep += (held.x - held.y) * weight;
            rates += rate;
            rises += (held.x - counted) * rate;
            level.shares[row][column] = weight;
        }
    }
    if total <= 0.0 {
        return level;
    }
    let share = 1.0 / total;
    let lift = raised * share;
    level.surface = counted + lift;
    level.slope = (rates * lift - rises) * (share / spacing);
    level.depth = max(deep * share, 0.0);
    level.held = total;
    level.held_rate = rates / spacing;
    level.shares = level.shares * share;
    return level;
}

fn water_read(cell: WaterCell) -> WaterAt {
    var level = water_level(cell);
    var at = water_at_level(level);
    if level.held <= 0.0 {
        return at;
    }
    let origin = water_texel(cell.block, cell.first - vec2<u32>(1u));
    let nearest = select(vec2<u32>(1u), vec2<u32>(2u), cell.within >= vec2<f32>(0.5));
    for (var row = 0u; row < 4u; row++) {
        for (var column = 0u; column < 4u; column++) {
            let share = level.shares[row][column];
            if share <= 0.0 {
                continue;
            }
            let steps = vec2<u32>(column, row);
            let words = water_words(textureLoad(water_info, origin.xy + vec2<i32>(steps),
                origin.z, 0));
            at = water_words_add(at, words, share, all(steps == nearest));
        }
    }
    return at;
}

fn water_at_level(level: WaterLevel) -> WaterAt {
    var at: WaterAt;
    at.surface = level.surface;
    at.slope = level.slope;
    at.depth = level.depth;
    at.held = level.held;
    return at;
}

fn water_words_add(at: WaterAt, words: WaterWords, share: f32, nearest: bool) -> WaterAt {
    var sum = at;
    sum.words.flow += words.flow * share;
    sum.words.span += words.span * share;
    sum.words.state += words.state * share;
    if nearest {
        sum.words.lake = words.lake;
        sum.words.sheet = words.sheet;
        sum.words.reach = words.reach;
    }
    return sum;
}

fn water_level_at(local: vec2<f32>) -> WaterLevel {
    return water_level(water_cell(local, 0u));
}
