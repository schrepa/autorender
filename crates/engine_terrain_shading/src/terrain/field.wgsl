import package::terrain::faces::heightfield::{TERRAIN_GRID, terrain};

@group(1) @binding(17) var terrain_heights: texture_2d_array<f32>;

struct PatchInput {
    @builtin(vertex_index) vertex: u32,
    @location(0) first: vec2<i32>,
    @location(1) shape: vec2<f32>,
    @location(2) atlas: vec2<u32>,
    @location(3) holes: u32,
}

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

fn spread_slopes(shortened: f32) -> f32 {
    let kept = clamp(shortened, 1.0e-4, 1.0);
    return 2.0 * (1.0 - kept) / kept;
}
