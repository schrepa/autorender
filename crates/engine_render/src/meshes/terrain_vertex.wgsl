// Terrain, the declarations every terrain pass shares: the height atlas, the
// views and each terrain's uniform; and the vertex stage of the patches the
// kept shadow levels draw the ground with, at a level of detail of their
// own: one shared grid of 17 x 17 vertices and a skirt, displaced by the
// heights of their tile's layer of the height atlas, unmorphed. Positions
// are relative to the terrain's eye, the camera's (camera-relative
// rendering): integer sample counts from an anchor on every level's grid,
// times the spacing, plus the anchor's offset from the eye. Skirts hang
// below patch edges and fill gaps where neighbors of other levels meet. The
// camera's ground draws its subdivision's leaves instead
// (`meshes/terrain_surface.wgsl`).

// Every tile's heights with a border of one sample, one layer each; only
// loaded.
@group(1) @binding(17) var terrain_heights: texture_2d_array<f32>;

// A view of the terrain: its projection of positions relative to its eye,
// jittered like the camera's on temporally antialiased frames, and the
// terrain's eye less this view's.
struct TerrainView {
    relative_view_projection: mat4x4<f32>,
    eye_offset: vec4<f32>,
    // Its inverse: eye-relative positions of depth-buffer points.
    inverse_relative_view_projection: mat4x4<f32>,
}
@group(1) @binding(18) var<uniform> terrain_view: TerrainView;

// One terrain: its anchor less its eye (w: the level-0 spacing), that eye's
// world position, its origin (w: 0), where its levels start in
// `terrain_directory` and how many there are, where its surface materials
// start in `terrain_palette`, and its layer of
// `terrain_sky` plus one, 0 until its lighting arrives.
struct TerrainUniform {
    anchor: vec4<f32>,
    eye: vec4<f32>,
    // Its subdivision's leaves: where they start in `terrain_surface_leaves`
    // (x), where each layer's word of `terrain_holes` naming its tile's hole
    // bits starts (y), and its anchor on the coarsest level's grid, in that
    // level's samples from its origin (z and w, signed).
    surface: vec4<u32>,
    origin: vec4<f32>,
    directory: vec4<u32>,
    // The ground's relief: the deepest any of its materials with maps
    // sinks it (meters) and the most steps its rays take; the deepest any of
    // them displaces it (meters), and the meters apart the vertices of its
    // subdivision's finest leaves stand, which displacement reads its height
    // maps no finer than.
    relief: vec4<f32>,
    // Runtime deformation's map: its first texel (world
    // texel indices along x and z), and its texels and blocks along a side
    // (0 while off); texels per meter, what is left of a track since the
    // map's reference time, and the rim's height for a track's depth.
    deform_grid: vec4<i32>,
    deform: vec4<f32>,
    // Ground cover's radius and the meters from the eye at which its blades
    // past the radius span the settings' pixels on this terrain
    // (`ground_cover::cover_span`), whose law the canopy follows as blades
    // retire (a radius of 0 shows it from the eye on), and its density
    // scale; all 0 without cover.
    cover: vec4<f32>,
    // Its runtime virtual texture (`terrain.wgsl`): its index
    // among the textured terrains, its mips (0 without one), slots along a
    // sheet's side, and this frame's feedback pixel (0 to 63) of each 8x8
    // block; the meters a texel of mip 0 spans, the inverse of a sheet's
    // texels along a side and the frame's time in seconds, which pages fade
    // in by from their arrivals; and where its windows and tables start in
    // `terrain_directory`.
    vt: vec4<u32>,
    vt_scale: vec4<f32>,
    vt_layout: vec4<u32>,
    // Its macro colors' layer of `terrain_macro` plus one (0 until they
    // arrive).
    macro_map: vec4<u32>,
    // Its surface's steep layer, which fragments decide at every point of
    // the ground (`engine_terrain::Steep`): its material plus one (0
    // without one) and the tangents of the slopes its share rises between;
    // and w 1 where its coarser levels shade with level 0's shading
    // averaged (`engine_terrain::TileShading`), whose steep material is
    // decided at level 0's samples alone, 0 for surroundings.
    steep: vec4<f32>,
    // The pixels a level's samples span at least where it shades the ground
    // whole, as its tiles stream (`TerrainSettings::pixel_error`):
    // `terrain_shading_level` hands each level over as they shrink to half
    // that (x).
    pixel_error: vec4<f32>,
    // Its tiles' water (`meshes/water.wgsl`): where water's directory
    // starts in `terrain_directory`, after the virtual texture's windows
    // and tables (x).
    water: vec4<u32>,
}
@group(1) @binding(19) var<uniform> terrain: TerrainUniform;

// A patch: its first sample counted from the anchor at its level; its
// spacing and its skirt's depth; its tile's layer and its first texel there
// (x | z << 16), borders aside; and where its tile's hole bits start in
// `terrain_holes` (0xffffffff without holes).
struct PatchInput {
    @builtin(vertex_index) vertex: u32,
    @location(0) first: vec2<i32>,
    @location(1) shape: vec2<f32>,
    @location(2) atlas: vec2<u32>,
    @location(3) holes: u32,
}

// Vertices along a patch side, and the grid's vertices before the skirts'.
const TERRAIN_GRID: u32 = 17u;
const TERRAIN_GRID_VERTICES: u32 = 289u;

fn terrain_load(layer: u32, texel: vec2<u32>) -> f32 {
    return textureLoad(terrain_heights, vec2<i32>(texel), i32(layer), 0).x;
}

// A patch's vertex relative to the terrain's eye: its place on the grid at
// its tile's height there, skirts hanging their depth below the grid's
// edges.
fn terrain_vertex(input: PatchInput) -> vec3<f32> {
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
    let texel = vec2<u32>(input.atlas.y & 0xffffu, input.atlas.y >> 16u) + grid + 1u;
    let height = terrain_load(input.atlas.x, texel);
    let base = vec2<f32>(input.first + vec2<i32>(grid));
    return vec3<f32>(base.x * spacing + terrain.anchor.x,
        height + terrain.anchor.y - select(0.0, input.shape.y, skirt),
        base.y * spacing + terrain.anchor.z);
}
