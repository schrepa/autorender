// The ground's true surface (`meshes/terrain_surface.wgsl`'s
// `terrain_true_height`), which the terrain's subdivision draws within about
// a pixel, for what stands on it: ground cover's blades (`cover_grow.wgsl`)
// and placed objects (`scatter_snap.wgsl`); and for what reflections' world
// trace meets (`reflections.wgsl`). Prepended to each, which declares the
// terrains' `directory` and `heights`.

const NOT_RESIDENT: u32 = 0xffffffffu;

// A terrain as its ground reads: where its levels start in `directory` and
// how many there are, and its level-0 sample spacing.
struct GroundTerrain {
    directory: u32,
    levels: u32,
    spacing: f32,
}

// A tile of a level: its layer (`NOT_RESIDENT` when not loaded) and its
// first sample (the level's samples from the terrain's origin).
struct LevelTile {
    layer: u32,
    first: vec2<u32>,
}

// The tile of `terrain`'s `level` holding its sample `samples`
// (terrain-local).
fn level_tile(terrain: GroundTerrain, level: u32, samples: vec2<f32>) -> LevelTile {
    let header = terrain.directory + level * 4u;
    let tiles = vec2<u32>(directory[header + 1u], directory[header + 2u]);
    let tile = min(vec2<u32>(max(samples, vec2<f32>(0.0)) / 256.0), tiles - vec2<u32>(1u));
    var found: LevelTile;
    found.layer = directory[directory[header] + tile.y * tiles.x + tile.x];
    found.first = tile * 256u;
    return found;
}

// The height of `tile`'s sample `sample` (its level's, terrain-local);
// layers keep a border sample before the tile's first.
fn tile_height(tile: LevelTile, sample: vec2<u32>) -> f32 {
    let texel = min(sample - tile.first, vec2<u32>(256u)) + vec2<u32>(1u);
    return textureLoad(heights, texel, tile.layer, 0).r;
}

// The height within a cell of corners `a` (0, 0), `b` (1, 0), `c` (0, 1)
// and `d` (1, 1) at `f`, on the two triangles every consumer splits cells
// into, from the (0, 1) corner to the (1, 0) one.
fn triangle_height(a: f32, b: f32, c: f32, d: f32, f: vec2<f32>) -> f32 {
    if f.x + f.y <= 1.0 {
        return a + (b - a) * f.x + (c - a) * f.y;
    }
    return d + (c - d) * (1.0 - f.x) + (b - d) * (1.0 - f.y);
}

// The ground's true height above `terrain`'s origin at terrain-local
// `local`, where its finest resident level is `level` (its tile `tile`): on
// that level's triangles, as the terrain's passes read it.
fn true_height(terrain: GroundTerrain, local: vec2<f32>, level: u32, tile: LevelTile) -> f32 {
    let samples = max(local / (terrain.spacing * f32(1u << level)), vec2<f32>(0.0));
    let within = clamp(samples - vec2<f32>(tile.first), vec2<f32>(0.0), vec2<f32>(256.0));
    let cell = min(vec2<u32>(within), vec2<u32>(255u));
    let corner = tile.first + cell;
    return triangle_height(tile_height(tile, corner), tile_height(tile, corner + vec2<u32>(1u, 0u)),
        tile_height(tile, corner + vec2<u32>(0u, 1u)), tile_height(tile, corner + vec2<u32>(1u, 1u)),
        within - vec2<f32>(cell));
}
