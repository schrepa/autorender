// Terrain (decision 051), the vertex stage every terrain pass shares: patches of one shared grid of 17 x 17 vertices
// and a skirt, displaced by the heights of their tile's layer of the height
// atlas. Each vertex morphs toward the next coarser level's vertex over the
// last quarter of its level's range (continuous distance-dependent level of
// detail): odd grid vertices slide onto their even neighbors and heights
// blend into the coarser tile's, so a patch fully morphed at its range's end
// matches its coarser neighbors exactly. Positions are relative to the eye
// the terrain streamed for (camera-relative rendering): integer sample
// counts from an anchor on every level's grid, times the spacing, plus the
// anchor's offset from the eye. Skirts hang below patch edges and fill gaps
// where a finer tile is not resident yet.

// Every tile's heights with a border of one sample, one layer each; only
// loaded.
@group(1) @binding(17) var terrain_heights: texture_2d_array<f32>;

// A view of the terrain: its projection of positions relative to its eye,
// jittered like the camera's on temporally antialiased frames, and the
// terrain's eye less this view's.
struct TerrainView {
    relative_view_projection: mat4x4<f32>,
    eye_offset: vec4<f32>,
}
@group(1) @binding(18) var<uniform> terrain_view: TerrainView;

// One terrain: its anchor less its eye (w: the level-0 spacing), that eye's
// world position, its origin, and its material (decision 051): linear grass,
// dirt, rock and snow colors; grass's w its roughness, rock's the cosine of
// the slope rock takes over at, snow's the height snow starts at above the
// origin; and where its levels start in `terrain_directory` and how many
// there are.
struct TerrainUniform {
    anchor: vec4<f32>,
    eye: vec4<f32>,
    origin: vec4<f32>,
    grass: vec4<f32>,
    dirt: vec4<f32>,
    rock: vec4<f32>,
    snow: vec4<f32>,
    directory: vec4<u32>,
}
@group(1) @binding(19) var<uniform> terrain: TerrainUniform;

// A patch: its first sample counted from the anchor at its level; its
// spacing, the distances its vertices morph over and its skirt's depth; its
// tile's layer, its parent tile's layer, and its first texel in each (x |
// z << 16), borders aside; and where its tile's hole bits start in
// `terrain_holes` (0xffffffff without holes).
struct PatchInput {
    @builtin(vertex_index) vertex: u32,
    @location(0) first: vec2<i32>,
    @location(1) shape: vec4<f32>,
    @location(2) atlas: vec4<u32>,
    @location(3) holes: u32,
}

// Vertices along a patch side, and the grid's vertices before the skirts'.
const TERRAIN_GRID: u32 = 17u;
const TERRAIN_GRID_VERTICES: u32 = 289u;

fn terrain_load(layer: u32, texel: vec2<u32>) -> f32 {
    return textureLoad(terrain_heights, vec2<i32>(texel), i32(layer), 0).x;
}

// A vertex relative to the terrain's eye, and where it lies in its tile's
// cells. Fragments take their normals from the terrain's normal maps, not
// from the level drawn (decision 051).
struct TerrainVertex {
    relative: vec3<f32>,
    cell: vec2<f32>,
}

fn terrain_vertex(input: PatchInput) -> TerrainVertex {
    var grid: vec2<u32>;
    var skirt = false;
    if input.vertex < TERRAIN_GRID_VERTICES {
        grid = vec2<u32>(input.vertex % TERRAIN_GRID, input.vertex / TERRAIN_GRID);
    } else {
        // Skirts along z = 0, z = 16, x = 0 and x = 16.
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
    let layer = input.atlas.x;
    let texel = vec2<u32>(input.atlas.z & 0xffffu, input.atlas.z >> 16u) + grid + 1u;
    let fine = terrain_load(layer, texel);
    let base = vec2<f32>(input.first + vec2<i32>(grid));
    let unmorphed = vec3<f32>(base.x * spacing + terrain.anchor.x, fine + terrain.anchor.y,
        base.y * spacing + terrain.anchor.z);
    let morph = clamp((length(unmorphed) - input.shape.y) / (input.shape.z - input.shape.y),
        0.0, 1.0);
    let odd = grid & vec2<u32>(1u);
    let parent_texel = vec2<u32>(input.atlas.w & 0xffffu, input.atlas.w >> 16u)
        + (grid - odd) / 2u + 1u;
    let coarse = terrain_load(input.atlas.y, parent_texel);
    // Weights summing exactly: fully morphed vertices take the coarser
    // tile's heights to the bit.
    let height = fine * (1.0 - morph) + coarse * morph;
    let morphed = base - vec2<f32>(odd) * morph;
    var vertex: TerrainVertex;
    vertex.relative = vec3<f32>(morphed.x * spacing + terrain.anchor.x,
        height + terrain.anchor.y - select(0.0, input.shape.w, skirt),
        morphed.y * spacing + terrain.anchor.z);
    vertex.cell = vec2<f32>(texel - 1u) + morphed - base;
    return vertex;
}

