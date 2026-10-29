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
