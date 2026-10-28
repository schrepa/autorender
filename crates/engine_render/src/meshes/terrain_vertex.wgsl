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
