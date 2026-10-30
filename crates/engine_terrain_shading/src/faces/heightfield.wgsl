import package::face::ground::NOT_RESIDENT;
import package::terrain::draw::{TerrainBlend, TerrainShadingLevel, terrain_band_rise,
    terrain_border_blend, terrain_control, terrain_control_sampler, terrain_controls_blend,
    terrain_gradient_between, terrain_layer_fades, terrain_material, terrain_material_rises,
    terrain_normals, terrain_steep_blend};
import package::terrain::field::{SURFACE_MESHLET_LEVEL, SURFACE_MESHLET_VERTICES,
    SurfaceTriangle, TerrainLevelHeight, TerrainUniform, TerrainView, surface_along,
    terrain_directory, terrain_load};
import package::terrain::water_blocks::{WATER_BLOCK, WATER_BLOCKS_SIDE, WATER_FLOW_RANGE,
    WATER_SHEAR_RANGE, WATER_TILE_WORDS, WaterLevel, WaterWords, water_extent, water_level,
    water_span, water_spline};

@group(1) @binding(18) var<uniform> terrain_view: TerrainView;

@group(1) @binding(19) var<uniform> terrain: TerrainUniform;

const TERRAIN_GRID: u32 = 17u;

@group(1) @binding(36) var<storage, read> terrain_surface_leaves: array<SurfaceTriangle>;

fn surface_vertex(triangle: SurfaceTriangle, vertex: u32) -> vec2<f32> {
    let parts = 1u << SURFACE_MESHLET_LEVEL;
    var row = parts;
    var first = SURFACE_MESHLET_VERTICES - parts - 1u;
    while vertex < first {
        first -= row;
        row--;
    }
    let j = vertex - first;
    let i = row - j;
    var tail = triangle.c;
    var head = triangle.b;
    var steps = j;
    if row == parts {
        tail = triangle.a;
    } else if j == 0u {
        head = triangle.a;
        steps = i;
    } else if i != 0u {
        let share = vec2<f32>(f32(i), f32(j)) / f32(parts);
        return triangle.c + (triangle.a - triangle.c) * share.x
            + (triangle.b - triangle.c) * share.y;
    }
    return surface_along(tail, head, steps);
}

fn terrain_level_height(local: vec2<f32>, level: u32) -> TerrainLevelHeight {
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    var held: TerrainLevelHeight;
    for (var index = level; index < levels; index++) {
        let header = base + index * 4u;
        let samples = max(local / (terrain.anchor.w * f32(1u << index)), vec2<f32>(0.0));
        let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
        let tile = min(vec2<u32>(samples / 256.0), tiles - vec2<u32>(1u));
        let layer = terrain_directory[terrain_directory[header] + tile.y * tiles.x + tile.x];
        if layer == 0xffffffffu {
            continue;
        }
        let within = clamp(samples - vec2<f32>(tile * 256u), vec2<f32>(0.0), vec2<f32>(256.0));
        let corner = min(vec2<u32>(within), vec2<u32>(255u));
        let f = within - vec2<f32>(corner);
        let texel = corner + vec2<u32>(1u);
        let h00 = terrain_load(layer, texel);
        let h10 = terrain_load(layer, texel + vec2<u32>(1u, 0u));
        let h01 = terrain_load(layer, texel + vec2<u32>(0u, 1u));
        let h11 = terrain_load(layer, texel + vec2<u32>(1u, 1u));
        held.found = true;
        if f.x + f.y <= 1.0 {
            held.height = h00 + f.x * (h10 - h00) + f.y * (h01 - h00);
        } else {
            held.height = h11 + (1.0 - f.x) * (h01 - h11) + (1.0 - f.y) * (h10 - h11);
        }
        return held;
    }
    return held;
}

struct TerrainSubdivision {
    square: vec4<f32>,
    tree: vec4<u32>,
    frame: vec4<u32>,
    extent: vec4<f32>,
    cutout: vec4<f32>,
    planes: array<vec4<f32>, 6>,
}

struct TerrainGroundSamples {
    controls: vec4<u32>,
    gradients: vec4<u32>,
    within: vec2<f32>,
}

fn terrain_ground_samples(local: vec2<f32>) -> TerrainGroundSamples {
    var samples = TerrainGroundSamples(vec4<u32>(0u), vec4<u32>(0u), vec2<f32>(0.0));
    let base = terrain.directory.x;
    let levels = terrain.directory.y;
    for (var index = 0u; index < levels; index++) {
        let header = base + index * 4u;
        let at = max(local / (terrain.anchor.w * f32(1u << index)), vec2<f32>(0.0));
        let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
        let tile = min(vec2<u32>(at / 256.0), tiles - vec2<u32>(1u));
        let layer = terrain_directory[terrain_directory[header] + tile.y * tiles.x + tile.x];
        if layer == 0xffffffffu || terrain_layer_fades(layer).y >= 1.0 {
            continue;
        }
        let within = clamp(at - vec2<f32>(tile * 256u), vec2<f32>(0.0), vec2<f32>(256.0));
        let corner = min(vec2<u32>(within), vec2<u32>(255u));
        let uv = (vec2<f32>(corner) + vec2<f32>(2.0)) / 259.0;
        samples.controls = textureGather(0, terrain_control, terrain_control_sampler, uv, layer);
        samples.gradients = textureGather(0, terrain_normals, terrain_control_sampler, uv, layer);
        samples.within = within - vec2<f32>(corner);
        return samples;
    }
    return samples;
}

struct TerrainGroundRise {
    rise: f32,
    relief: bool,
}

fn terrain_ground_rise(samples: TerrainGroundSamples, local: vec2<f32>, height: f32,
        meters: f32) -> TerrainGroundRise {
    let gradient = terrain_gradient_between(samples.gradients, samples.within);
    let normal = normalize(vec3<f32>(-gradient.x, 1.0, -gradient.y));
    var blend: TerrainBlend;
    blend.ids = vec4<u32>(0xffffffffu);
    blend.weights = vec4<f32>(0.0);
    terrain_controls_blend(&blend, samples.controls, samples.within, 1.0);
    terrain_steep_blend(&blend, normal);
    terrain_border_blend(&blend, local, TerrainShadingLevel(0u, 0.0));
    let world = terrain.origin.xyz + vec3<f32>(local.x, height, local.y);
    var raised = TerrainGroundRise(0.0, false);
    raised.rise = terrain_band_rise(blend, world, normal, vec2<f32>(meters)).x * normal.y;
    for (var slot = 0u; slot < 4u; slot++) {
        if blend.weights[slot] > 0.0 && terrain_material_rises(terrain_material(blend.ids[slot])) {
            raised.relief = true;
        }
    }
    return raised;
}

@group(1) @binding(51) var water_info: texture_2d_array<u32>;

const WATER_LIVE_BLOCK: u32 = 8u;

const NO_WATER_BLOCK: u32 = 0xffffffffu;
const NO_WATER_SURFACE: f32 = -1.0e30;
const WATER_TILE_PATCHES: u32 = 16u;
const WATER_SPAN_LEAST: f32 = 0.25;
const WATER_SPAN_MOST: f32 = 4096.0;

const WATER_HELD: f32 = 0.5;

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

struct WaterAt {
    surface: f32,
    depth: f32,
    held: f32,
    words: WaterWords,
    slope: vec2<f32>,
}

struct WaterCell {
    level: u32,
    block: u32,
    first: vec2<u32>,
    within: vec2<f32>,
}

fn water_cell(local: vec2<f32>, level: u32) -> WaterCell {
    let levels = terrain.directory.y;
    var cell: WaterCell;
    cell.level = levels;
    cell.block = NO_WATER_BLOCK;
    if any(local < vec2<f32>(0.0)) || any(local > water_extent()) {
        return cell;
    }
    let base = terrain.directory.x;
    let quads = TERRAIN_GRID - 1u;
    let tile_quads = WATER_TILE_PATCHES * quads;
    for (var index = level; index < levels; index++) {
        let header = base + index * 4u;
        let samples = local / (terrain.anchor.w * f32(1u << index));
        let tiles = vec2<u32>(terrain_directory[header + 1u], terrain_directory[header + 2u]);
        let tile = min(vec2<u32>(samples / f32(tile_quads)), tiles - vec2<u32>(1u));
        let layer = terrain_directory[terrain_directory[header] + tile.y * tiles.x + tile.x];
        if layer == NOT_RESIDENT {
            continue;
        }
        let within = clamp(samples - vec2<f32>(tile * tile_quads), vec2<f32>(0.0),
            vec2<f32>(f32(tile_quads)));
        let corner = min(vec2<u32>(within), vec2<u32>(tile_quads - 1u));
        let tile_patch = corner / quads;
        cell.level = index;
        cell.block = terrain_directory[terrain.water.x + layer * WATER_TILE_WORDS
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
