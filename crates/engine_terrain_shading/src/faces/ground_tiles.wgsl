import package::place::ground_tiles::{GroundHeight, GroundMaterial, GroundTerrain, GroundTile,
    LevelTile, MaterialCover, MaterialLook, TerrainInfo};

const NOT_RESIDENT: u32 = 0xffffffffu;
const TILE_QUADS: u32 = 256u;
const PEAK_MIPS: u32 = 8u;
const PEAK_CELLS: u32 = 128u;

struct PaletteEntry {
    color: vec4<f32>,
    params: vec4<f32>,
    maps: vec4<f32>,
    far: vec4<f32>,
    relief: vec4<f32>,
    canopy: vec4<f32>,
    placed: vec4<f32>,
    placed_size: vec4<f32>,
    heights: vec4<f32>,
}

const PALETTE_ENTRIES: u32 = 113u;

@group(0) @binding(0) var heights: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read> directory: array<u32>;
@group(0) @binding(2) var control: texture_2d_array<u32>;
@group(0) @binding(3) var normals: texture_2d_array<u32>;
@group(0) @binding(4) var macro_maps: texture_2d_array<f32>;
@group(0) @binding(5) var<uniform> palette: array<PaletteEntry, PALETTE_ENTRIES>;
@group(0) @binding(6) var<storage, read> covers: array<MaterialCover>;

fn ground_of(info: TerrainInfo) -> GroundTerrain {
    return GroundTerrain(info.levels.x, info.levels.y, info.origin.w);
}

fn ground_extent(terrain: GroundTerrain) -> vec2<f32> {
    return vec2<f32>(f32(directory[terrain.directory + 1u]), f32(directory[terrain.directory + 2u]))
        * 256.0 * terrain.spacing;
}

fn level_tiles(terrain: GroundTerrain, level: u32) -> vec2<u32> {
    let header = terrain.directory + level * 4u;
    return vec2<u32>(directory[header + 1u], directory[header + 2u]);
}

fn tile_layer(terrain: GroundTerrain, level: u32, tile: vec2<u32>) -> u32 {
    let header = terrain.directory + level * 4u;
    let tiles = vec2<u32>(directory[header + 1u], directory[header + 2u]);
    return directory[directory[header] + tile.y * tiles.x + tile.x];
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

fn tile_peak(tile: LevelTile, texel: vec2<u32>, mip: u32) -> f32 {
    return textureLoad(heights, texel, tile.layer, mip + 1u).r;
}

fn tile_gradient(tile: LevelTile, sample: vec2<u32>) -> vec2<f32> {
    let texel = min(sample - tile.first, vec2<u32>(256u)) + vec2<u32>(1u);
    return unpack2x16float(textureLoad(normals, texel, tile.layer, 0).x);
}

fn tile_material(tile: LevelTile, sample: vec2<u32>) -> GroundMaterial {
    let texel = min(sample - tile.first, vec2<u32>(256u)) + vec2<u32>(1u);
    let value = textureLoad(control, texel, tile.layer, 0).x;
    return GroundMaterial(value & 0xffu, (value >> 8u) & 0xffu, (value >> 16u) & 0xffu);
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

fn ground_tile(terrain: GroundTerrain, local: vec2<f32>, first: u32) -> GroundTile {
    var found: GroundTile;
    found.tile.layer = NOT_RESIDENT;
    for (var level = first; level < terrain.levels; level++) {
        let tile = level_tile(terrain, level, local / (terrain.spacing * f32(1u << level)));
        if tile.layer != NOT_RESIDENT {
            found.found = true;
            found.level = level;
            found.tile = tile;
            return found;
        }
    }
    return found;
}

fn ground_height(terrain: GroundTerrain, local: vec2<f32>) -> GroundHeight {
    var ground: GroundHeight;
    let found = ground_tile(terrain, local, 0u);
    if found.found {
        ground.found = true;
        ground.level = found.level;
        ground.tile = found.tile;
        ground.height = true_height(terrain, local, found.level, found.tile);
    }
    return ground;
}

fn ground_colors_mip(terrain: GroundTerrain, footprint: f32) -> u32 {
    let extent = ground_extent(terrain);
    let finest = vec2<f32>(textureDimensions(macro_maps, 0));
    let texel = max(extent.x / finest.x, extent.y / finest.y);
    let coarsest = f32(textureNumLevels(macro_maps) - 1u);
    return u32(clamp(floor(log2(max(footprint / texel, 1.0))), 0.0, coarsest));
}

fn ground_colors(terrain: GroundTerrain, layer: u32, local: vec2<f32>, mip: u32) -> vec4<f32> {
    if layer == 0u {
        return vec4<f32>(0.5);
    }
    let extent = ground_extent(terrain);
    let size = vec2<i32>(textureDimensions(macro_maps, mip));
    let at = local / extent * vec2<f32>(size) - 0.5;
    let first = vec2<i32>(floor(at));
    let f = at - floor(at);
    let last = size - vec2<i32>(1);
    let a = textureLoad(macro_maps, clamp(first, vec2<i32>(0), last), layer - 1u, mip);
    let b = textureLoad(macro_maps, clamp(first + vec2<i32>(1, 0), vec2<i32>(0), last),
        layer - 1u, mip);
    let c = textureLoad(macro_maps, clamp(first + vec2<i32>(0, 1), vec2<i32>(0), last),
        layer - 1u, mip);
    let d = textureLoad(macro_maps, clamp(first + vec2<i32>(1, 1), vec2<i32>(0), last),
        layer - 1u, mip);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn macro_color(colors: vec4<f32>) -> vec3<f32> {
    return colors.rgb * (255.0 / 128.0);
}

fn material_look(base: u32, material: u32) -> MaterialLook {
    let entry = palette[min(base + material, PALETTE_ENTRIES - 1u)];
    return MaterialLook(entry.far, entry.canopy);
}

fn material_cover(base: u32, material: u32) -> MaterialCover {
    return covers[min(base + material, PALETTE_ENTRIES - 1u)];
}
