import package::face::ground_tiles::{GroundTerrain, NOT_RESIDENT, level_tiles, tile_layer};
import package::face::heightfield::{TERRAIN_GRID, terrain_uniform};
import package::generated::water::{NO_WATER_BLOCK, NO_WATER_SURFACE, WATER_BLOCK, WATER_BLOCKS_SIDE,
    WATER_FLOW_RANGE, WATER_SHEAR_RANGE, WATER_SPAN_LEAST, WATER_SPAN_MOST, WATER_SPAN_STEPS,
    WATER_TILE_PATCHES, WATER_TILE_WORDS, WATER_WAITING, water_state_word_reach};
import package::place::water::{WaterAt, WaterCell, WaterLevel, WaterWords};

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
    return select(WATER_SPAN_LEAST * exp2(octaves * (steps - 1.0) / WATER_SPAN_STEPS), 0.0,
        steps < 0.5);
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
    let held = array<vec2<f32>, 16>(
        water_held(origin, 0, 0), water_held(origin, 1, 0), water_held(origin, 2, 0),
        water_held(origin, 3, 0), water_held(origin, 0, 1), water_held(origin, 1, 1),
        water_held(origin, 2, 1), water_held(origin, 3, 1), water_held(origin, 0, 2),
        water_held(origin, 1, 2), water_held(origin, 2, 2), water_held(origin, 3, 2),
        water_held(origin, 0, 3), water_held(origin, 1, 3), water_held(origin, 2, 3),
        water_held(origin, 3, 3));
    return water_smooth(held, cell.within, terrain_uniform().anchor.w * f32(1u << cell.level));
}

fn water_held(origin: vec3<i32>, column: i32, row: i32) -> vec2<f32> {
    return textureLoad(water_surface, origin.xy + vec2<i32>(column, row), origin.z, 0).xy;
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
    read.reach = water_state_word_reach(words.y);
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
        let block = water_directory[layer * WATER_TILE_WORDS
            + tile_patch.y * WATER_TILE_PATCHES + tile_patch.x];
        if block == WATER_WAITING {
            continue;
        }
        cell.level = index;
        cell.block = block;
        cell.first = corner - tile_patch * quads + vec2<u32>(1u);
        cell.within = within - vec2<f32>(corner);
        return cell;
    }
    return cell;
}

fn water_height(held: vec2<f32>) -> f32 {
    return select(held.y, held.x, held.x > NO_WATER_SURFACE * 0.5);
}

struct WaterSmoothing {
    total: f32,
    raised: f32,
    rises: vec2<f32>,
    held: f32,
    depth: f32,
    wet_rates: vec2<f32>,
    share: f32,
    row: vec4<f32>,
}

fn water_smooth_add(sum: WaterSmoothing, held: vec2<f32>, weight: f32, rate: vec2<f32>,
        counted: f32) -> WaterSmoothing {
    var next = sum;
    let height = water_height(held) - counted;
    next.raised += height * weight;
    next.rises += height * rate;
    next.share = 0.0;
    if held.x <= NO_WATER_SURFACE * 0.5 {
        return next;
    }
    next.total += weight;
    next.share = weight;
    if held.x > held.y {
        next.held += weight;
        next.depth += (held.x - held.y) * weight;
        next.wet_rates += rate;
    }
    return next;
}

fn water_smooth_row(sum: WaterSmoothing, first: vec2<f32>, second: vec2<f32>, third: vec2<f32>,
        fourth: vec2<f32>, along: WaterSpline, weight: f32, rate: f32, counted: f32)
        -> WaterSmoothing {
    var next = water_smooth_add(sum, first, along.weights.x * weight,
        vec2<f32>(along.rates.x * weight, along.weights.x * rate), counted);
    var row = vec4<f32>(next.share, 0.0, 0.0, 0.0);
    next = water_smooth_add(next, second, along.weights.y * weight,
        vec2<f32>(along.rates.y * weight, along.weights.y * rate), counted);
    row.y = next.share;
    next = water_smooth_add(next, third, along.weights.z * weight,
        vec2<f32>(along.rates.z * weight, along.weights.z * rate), counted);
    row.z = next.share;
    next = water_smooth_add(next, fourth, along.weights.w * weight,
        vec2<f32>(along.rates.w * weight, along.weights.w * rate), counted);
    row.w = next.share;
    next.row = row;
    return next;
}

fn water_smooth(surfaces: array<vec2<f32>, 16>, within: vec2<f32>, spacing: f32)
        -> WaterLevel {
    let along = water_spline(within.x);
    let across = water_spline(within.y);
    let counted = water_height(surfaces[5]);
    var sum: WaterSmoothing;
    sum = water_smooth_row(sum, surfaces[0], surfaces[1], surfaces[2], surfaces[3], along,
        across.weights.x, across.rates.x, counted);
    let first = sum.row;
    sum = water_smooth_row(sum, surfaces[4], surfaces[5], surfaces[6], surfaces[7], along,
        across.weights.y, across.rates.y, counted);
    let second = sum.row;
    sum = water_smooth_row(sum, surfaces[8], surfaces[9], surfaces[10], surfaces[11], along,
        across.weights.z, across.rates.z, counted);
    let third = sum.row;
    sum = water_smooth_row(sum, surfaces[12], surfaces[13], surfaces[14], surfaces[15], along,
        across.weights.w, across.rates.w, counted);
    var level: WaterLevel;
    level.surface = counted + sum.raised;
    level.slope = -sum.rises / spacing;
    level.depth = sum.depth;
    level.held = sum.held;
    level.held_rate = sum.wet_rates / spacing;
    level.shares = mat4x4<f32>(first, second, third, sum.row);
    if sum.total > 0.0 {
        level.shares = level.shares * (1.0 / sum.total);
    }
    return level;
}

fn water_read(cell: WaterCell) -> WaterAt {
    let level = water_level(cell);
    var at = water_at_level(level);
    if level.held <= 0.0 {
        return at;
    }
    let origin = water_texel(cell.block, cell.first - vec2<u32>(1u));
    let nearest = select(vec2<u32>(1u), vec2<u32>(2u), cell.within >= vec2<f32>(0.5));
    at = water_read_row(at, origin, 0u, level.shares[0], nearest);
    at = water_read_row(at, origin, 1u, level.shares[1], nearest);
    at = water_read_row(at, origin, 2u, level.shares[2], nearest);
    return water_read_row(at, origin, 3u, level.shares[3], nearest);
}

fn water_read_row(at: WaterAt, origin: vec3<i32>, row: u32, shares: vec4<f32>,
        nearest: vec2<u32>) -> WaterAt {
    var sum = water_read_word(at, origin, vec2<u32>(0u, row), shares.x, nearest);
    sum = water_read_word(sum, origin, vec2<u32>(1u, row), shares.y, nearest);
    sum = water_read_word(sum, origin, vec2<u32>(2u, row), shares.z, nearest);
    return water_read_word(sum, origin, vec2<u32>(3u, row), shares.w, nearest);
}

fn water_read_word(at: WaterAt, origin: vec3<i32>, steps: vec2<u32>, share: f32,
        nearest: vec2<u32>) -> WaterAt {
    if share <= 0.0 {
        return at;
    }
    let words = water_words(textureLoad(water_info, origin.xy + vec2<i32>(steps), origin.z, 0));
    return water_words_add(at, words, share, all(steps == nearest));
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
