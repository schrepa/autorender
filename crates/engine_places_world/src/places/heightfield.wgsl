const TERRAIN_GRID: u32 = 17u;
const SURFACE_LEAF_INDICES: u32 = 0u;

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

fn use_terrain(index: u32) {
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

fn surface_leaf_count() -> u32 {
    return 0u;
}

fn bed_height(at: vec2<f32>) -> f32 {
    return 0.0;
}
