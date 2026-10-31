const TERRAIN_GRID: u32 = 17u;
const WATER_LIVE_BLOCK: u32 = 8u;
const NO_WATER_BLOCK: u32 = 0xffffffffu;
const NO_WATER_SURFACE: f32 = -1.0e30;
const WATER_TILE_PATCHES: u32 = 16u;
const WATER_SPAN_LEAST: f32 = 0.25;
const WATER_SPAN_MOST: f32 = 4096.0;
const WATER_HELD: f32 = 0.5;

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

struct TerrainView {
    relative_view_projection: mat4x4<f32>,
    eye_offset: vec4<f32>,
    inverse_relative_view_projection: mat4x4<f32>,
}

struct SurfaceTriangle {
    a: vec2<f32>,
    b: vec2<f32>,
    c: vec2<f32>,
}

struct TerrainLevelHeight {
    height: f32,
    found: bool,
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

struct TerrainGroundRise {
    rise: f32,
    relief: bool,
}

struct WaterWords {
    flow: vec2<f32>,
    span: f32,
    lake: bool,
    sheet: bool,
    state: vec3<f32>,
    reach: u32,
}

struct WaterLevel {
    surface: f32,
    slope: vec2<f32>,
    depth: f32,
    held: f32,
    held_rate: vec2<f32>,
    shares: mat4x4<f32>,
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

fn terrain_uniform() -> TerrainUniform {
    var found: TerrainUniform;
    return found;
}

fn terrain_view_uniform() -> TerrainView {
    var found: TerrainView;
    return found;
}

fn surface_leaf(index: u32) -> SurfaceTriangle {
    var leaf: SurfaceTriangle;
    return leaf;
}

fn water_info_at(texel: vec3<i32>) -> vec4<u32> {
    return vec4<u32>(0u);
}

fn surface_vertex(triangle: SurfaceTriangle, vertex: u32) -> vec2<f32> {
    return vec2<f32>(0.0);
}

fn terrain_level_height(local: vec2<f32>, level: u32) -> TerrainLevelHeight {
    var height: TerrainLevelHeight;
    return height;
}

fn terrain_ground_samples(local: vec2<f32>) -> TerrainGroundSamples {
    var samples: TerrainGroundSamples;
    return samples;
}

fn terrain_ground_rise(samples: TerrainGroundSamples, local: vec2<f32>, height: f32,
        meters: f32) -> TerrainGroundRise {
    var rise: TerrainGroundRise;
    return rise;
}

fn water_texel(block: u32, sample: vec2<u32>) -> vec3<i32> {
    return vec3<i32>(0);
}

fn water_words(words: vec4<u32>) -> WaterWords {
    var found: WaterWords;
    return found;
}

fn water_cell(local: vec2<f32>, level: u32) -> WaterCell {
    var cell: WaterCell;
    cell.block = NO_WATER_BLOCK;
    return cell;
}

fn water_smooth(surfaces: array<vec2<f32>, 16>, within: vec2<f32>, spacing: f32) -> WaterLevel {
    var level: WaterLevel;
    level.surface = NO_WATER_SURFACE;
    return level;
}

fn water_read(cell: WaterCell) -> WaterAt {
    var at: WaterAt;
    at.surface = NO_WATER_SURFACE;
    return at;
}

fn water_at_level(level: WaterLevel) -> WaterAt {
    var at: WaterAt;
    at.surface = NO_WATER_SURFACE;
    return at;
}

fn water_words_add(at: WaterAt, words: WaterWords, share: f32, nearest: bool) -> WaterAt {
    return at;
}
