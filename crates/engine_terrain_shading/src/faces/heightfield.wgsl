import package::terrain::draw::{TerrainBlend, TerrainShadingLevel, terrain_band_rise,
    terrain_border_blend, terrain_control, terrain_control_sampler, terrain_controls_blend,
    terrain_gradient_between, terrain_layer_fades, terrain_material, terrain_material_rises,
    terrain_normals, terrain_steep_blend};

const NOT_RESIDENT: u32 = 0xffffffffu;

@group(0) @binding(0) var heights: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> directory: array<u32>;

struct GroundTerrain {
    directory: u32,
    levels: u32,
    spacing: f32,
}

struct LevelTile {
    layer: u32,
    first: vec2<u32>,
}

fn level_tile(terrain: GroundTerrain, level: u32, samples: vec2<f32>) -> LevelTile {
    let header = terrain.directory + level * 4u;
    let tiles = vec2<u32>(directory[header + 1u], directory[header + 2u]);
    let tile = min(vec2<u32>(max(samples, vec2<f32>(0.0)) / 256.0), tiles - vec2<u32>(1u));
    var found: LevelTile;
    found.layer = directory[directory[header] + tile.y * tiles.x + tile.x];
    found.first = tile * 256u;
    return found;
}

fn tile_height(tile: LevelTile, sample: vec2<u32>) -> f32 {
    let texel = min(sample - tile.first, vec2<u32>(256u)) + vec2<u32>(1u);
    return textureLoad(heights, texel, tile.layer, 0).r;
}

fn triangle_height(a: f32, b: f32, c: f32, d: f32, f: vec2<f32>) -> f32 {
    if f.x + f.y <= 1.0 {
        return a + (b - a) * f.x + (c - a) * f.y;
    }
    return d + (c - d) * (1.0 - f.x) + (b - d) * (1.0 - f.y);
}

fn true_height(terrain: GroundTerrain, local: vec2<f32>, level: u32, tile: LevelTile) -> f32 {
    let samples = max(local / (terrain.spacing * f32(1u << level)), vec2<f32>(0.0));
    let within = clamp(samples - vec2<f32>(tile.first), vec2<f32>(0.0), vec2<f32>(256.0));
    let cell = min(vec2<u32>(within), vec2<u32>(255u));
    let corner = tile.first + cell;
    return triangle_height(tile_height(tile, corner), tile_height(tile, corner + vec2<u32>(1u, 0u)),
        tile_height(tile, corner + vec2<u32>(0u, 1u)), tile_height(tile, corner + vec2<u32>(1u, 1u)),
        within - vec2<f32>(cell));
}

@group(1) @binding(17) var terrain_heights: texture_2d_array<f32>;

struct TerrainView {
    relative_view_projection: mat4x4<f32>,
    eye_offset: vec4<f32>,
    inverse_relative_view_projection: mat4x4<f32>,
}
@group(1) @binding(18) var<uniform> terrain_view: TerrainView;

struct TerrainUniform {
    anchor: vec4<f32>,
    eye: vec4<f32>,
    surface: vec4<u32>,
    origin: vec4<f32>,
    directory: vec4<u32>,
    relief: vec4<f32>,
    cover: vec4<f32>,
    macro_map: vec4<u32>,
    steep: vec4<f32>,
    pixel_error: vec4<f32>,
    water: vec4<u32>,
}
@group(1) @binding(19) var<uniform> terrain: TerrainUniform;

struct PatchInput {
    @builtin(vertex_index) vertex: u32,
    @location(0) first: vec2<i32>,
    @location(1) shape: vec2<f32>,
    @location(2) atlas: vec2<u32>,
    @location(3) holes: u32,
}

const TERRAIN_GRID: u32 = 17u;
const TERRAIN_GRID_VERTICES: u32 = 289u;

fn terrain_load(layer: u32, texel: vec2<u32>) -> f32 {
    return textureLoad(terrain_heights, vec2<i32>(texel), i32(layer), 0).x;
}

fn terrain_vertex(input: PatchInput) -> vec3<f32> {
    var grid: vec2<u32>;
    var skirt = false;
    if input.vertex < TERRAIN_GRID_VERTICES {
        grid = vec2<u32>(input.vertex % TERRAIN_GRID, input.vertex / TERRAIN_GRID);
    } else {
        let index = input.vertex - TERRAIN_GRID_VERTICES;
        let edge = index / TERRAIN_GRID;
        let along = index % TERRAIN_GRID;
        let last = TERRAIN_GRID - 1u;
        let across_z = vec2<u32>(along, select(last, 0u, edge == 0u));
        let across_x = vec2<u32>(select(last, 0u, edge == 2u), along);
        grid = select(across_z, across_x, edge >= 2u);
        skirt = true;
    }
    let spacing = input.shape.x;
    let texel = vec2<u32>(input.atlas.y & 0xffffu, input.atlas.y >> 16u) + grid + 1u;
    let height = terrain_load(input.atlas.x, texel);
    let base = vec2<f32>(input.first + vec2<i32>(grid));
    return vec3<f32>(base.x * spacing + terrain.anchor.x,
        height + terrain.anchor.y - select(0.0, input.shape.y, skirt),
        base.y * spacing + terrain.anchor.z);
}

@group(1) @binding(23) var<storage, read> terrain_directory: array<u32>;

struct SurfaceTriangle {
    a: vec2<f32>,
    b: vec2<f32>,
    c: vec2<f32>,
}

@group(1) @binding(36) var<storage, read> terrain_surface_leaves: array<SurfaceTriangle>;

const SURFACE_MESHLET_LEVEL: u32 = 2u;
const SURFACE_MESHLET_VERTICES: u32 = 15u;
const SURFACE_MESHLET_INDICES: u32 = 48u;

fn surface_along(tail: vec2<f32>, head: vec2<f32>, steps: u32) -> vec2<f32> {
    var low = tail;
    var high = head;
    for (var bit = SURFACE_MESHLET_LEVEL; bit > 0u; bit--) {
        let middle = (low + high) * 0.5;
        if ((steps >> (bit - 1u)) & 1u) == 0u {
            high = middle;
        } else {
            low = middle;
        }
    }
    return select(low, head, steps == (1u << SURFACE_MESHLET_LEVEL));
}

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

struct TerrainLevelHeight {
    height: f32,
    found: bool,
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

const COVER_LUSH_RATIO: f32 = 2.0;

const COVER_LUSH_REFLECTANCE: vec3<f32> = vec3<f32>(0.67, 0.8, 0.89);

struct CoverGrowth {
    height: f32,
    density: f32,
    tint: vec3<f32>,
}

fn cover_growth(lush: f32) -> CoverGrowth {
    let toward = saturate(lush) - 0.5;
    var growth: CoverGrowth;
    growth.height = pow(COVER_LUSH_RATIO, toward);
    growth.density = growth.height;
    growth.tint = pow(COVER_LUSH_REFLECTANCE, vec3<f32>(toward));
    return growth;
}

fn macro_tint(colors: vec4<f32>, vegetated: f32) -> vec3<f32> {
    let growth = cover_growth(colors.a);
    return colors.rgb * (255.0 / 128.0) * mix(vec3<f32>(1.0), growth.tint, vegetated);
}

const COVER_CLUMP_LATTICES: vec2<f32> = vec2<f32>(0.4, 1.6);

const COVER_CLUMP_HEIGHT: f32 = 0.4;
const COVER_CLUMP_DRY: f32 = 0.4;
const COVER_CLUMP_HUE: f32 = 0.1;
const COVER_CLUMP_BRIGHT: f32 = 0.3;
const COVER_CLUMP_LEAN: f32 = 0.35;

fn cover_clump_spacing(layer: u32) -> f32 {
    return select(COVER_CLUMP_LATTICES.x, COVER_CLUMP_LATTICES.y, (layer & 1u) == 1u);
}

fn cover_clump_hash(cell: vec2<i32>, layer: u32) -> vec4<f32> {
    var h = (u32(cell.x) * 0x8da6b343u) ^ (u32(cell.y) * 0xd8163841u)
        ^ ((layer + 1u) * 0x2c1b3c6du);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return unpack4x8unorm(h ^ (h >> 16u));
}

fn cover_clump_layer(at: vec2<f32>, layer: u32) -> vec4<f32> {
    let p = at / cover_clump_spacing(layer);
    let cell = vec2<i32>(floor(p));
    let f = p - floor(p);
    let s = f * f * (3.0 - 2.0 * f);
    let a = cover_clump_hash(cell, layer);
    let b = cover_clump_hash(cell + vec2<i32>(1, 0), layer);
    let c = cover_clump_hash(cell + vec2<i32>(0, 1), layer);
    let d = cover_clump_hash(cell + vec2<i32>(1, 1), layer);
    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y) * 2.0 - 1.0;
}

fn cover_clump(at: vec2<f32>) -> vec4<f32> {
    return 0.5 * (cover_clump_layer(at, 0u) + cover_clump_layer(at, 1u));
}

fn cover_clump_lean(at: vec2<f32>) -> vec2<f32> {
    let layers = cover_clump_layer(at, 2u).xy + cover_clump_layer(at, 3u).xy;
    return 0.5 * COVER_CLUMP_LEAN * layers;
}

struct CoverClumped {
    height: f32,
    tint: vec3<f32>,
}

fn cover_clumped(look: vec4<f32>) -> CoverClumped {
    var clumped: CoverClumped;
    clumped.height = 1.0 + COVER_CLUMP_HEIGHT * look.x;
    let dry = pow(COVER_LUSH_REFLECTANCE, vec3<f32>(-COVER_CLUMP_DRY * look.y));
    let hue = vec3<f32>(1.0 + COVER_CLUMP_HUE * look.z, 1.0, 1.0 - COVER_CLUMP_HUE * look.z);
    clumped.tint = dry * hue * (1.0 + COVER_CLUMP_BRIGHT * look.w);
    return clumped;
}

fn spread_slopes(shortened: f32) -> f32 {
    let kept = clamp(shortened, 1.0e-4, 1.0);
    return 2.0 * (1.0 - kept) / kept;
}

const COVER_LEAVES: u32 = 5u;
const COVER_LEAF_STEPS: u32 = 255u / COVER_LEAVES;

const COVER_LEAF_ASPECT: f32 = 0.1;

const COVER_FAR_SIDE: f32 = 0.8;
const COVER_ROOT_OCCLUSION: f32 = 0.35;
const COVER_ROUGHNESS: f32 = 0.7;

fn seen_share(pixels: f32) -> f32 {
    return saturate(1.0 + log2(max(pixels, 1.0e-6)));
}

struct CoverLaw {
    span: f32,
}

fn cover_widening(away: f32, law: CoverLaw) -> f32 {
    return clamp(away / max(law.span, 1.0e-6), 1.0, f32(COVER_LEAVES));
}

fn cover_tuft(away: f32, height: f32, root: f32, law: CoverLaw) -> vec2<f32> {
    let tuft = root * cover_widening(away, law);
    let leaf = min(tuft, max(root, COVER_LEAF_ASPECT * height));
    return vec2<f32>(leaf, clamp(tuft / leaf, 1.0, f32(COVER_LEAVES)));
}

fn cover_share(away: f32, law: CoverLaw) -> f32 {
    let widening = cover_widening(away, law);
    return seen_share(law.span * widening / max(away, 1.0e-3)) / widening;
}

fn cover_shown(away: f32, share: f32, law: CoverLaw) -> f32 {
    return min(share * cover_widening(away, law), 1.0);
}

struct CoverSeen {
    tip: f32,
    occlusion: f32,
}

fn cover_seen(depth: f32) -> CoverSeen {
    let crossed = max(depth, 1.0e-3);
    let root = exp(-crossed);
    let low = exp(-0.465 * crossed);
    let middle = exp(-0.107 * crossed);
    let high = exp(-0.0274 * crossed);
    let met = 1.0 - root;
    var seen: CoverSeen;
    seen.tip = ((low - root) * 0.177 + (middle - low) * 0.786 + (1.0 - middle)) / met;
    let along = ((low - root) * 0.168 + (middle - low) * 0.507 + (high - middle) * 0.767
        + (1.0 - high) * 0.900) / met;
    seen.occlusion = mix(COVER_ROOT_OCCLUSION, 1.0, along);
    return seen;
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

@group(1) @binding(50) var water_surface: texture_2d_array<f32>;
@group(1) @binding(51) var water_info: texture_2d_array<u32>;

const WATER_LIVE_BLOCK: u32 = 8u;

const WATER_BLOCK: u32 = 19u;
const WATER_BLOCKS_SIDE: u32 = 16u;
const NO_WATER_BLOCK: u32 = 0xffffffffu;
const NO_WATER_SURFACE: f32 = -1.0e30;
const WATER_TILE_PATCHES: u32 = 16u;
const WATER_TILE_WORDS: u32 = 256u;
const WATER_FLOW_RANGE: f32 = 8.0;
const WATER_SPAN_LEAST: f32 = 0.25;
const WATER_SPAN_MOST: f32 = 4096.0;
const WATER_SHEAR_RANGE: f32 = 8.0;

const WATER_HELD: f32 = 0.5;

fn water_texel(block: u32, sample: vec2<u32>) -> vec3<i32> {
    let blocks = WATER_BLOCKS_SIDE * WATER_BLOCKS_SIDE;
    let index = block % blocks;
    let origin = vec2<u32>(index % WATER_BLOCKS_SIDE, index / WATER_BLOCKS_SIDE) * WATER_BLOCK;
    return vec3<i32>(vec2<i32>(origin + sample), i32(block / blocks));
}

struct WaterWords {
    flow: vec2<f32>,
    span: f32,
    lake: bool,
    sheet: bool,
    state: vec3<f32>,
    reach: u32,
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

struct WaterLevel {
    surface: f32,
    slope: vec2<f32>,
    depth: f32,
    held: f32,
    held_rate: vec2<f32>,
    shares: mat4x4<f32>,
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
