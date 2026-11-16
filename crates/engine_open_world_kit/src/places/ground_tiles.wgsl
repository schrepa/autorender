const NOT_RESIDENT: u32 = 0xffffffffu;
const PEAK_MIPS: u32 = 8u;
const PEAK_CELLS: u32 = 128u;

struct GroundTerrain {
    directory: u32,
    levels: u32,
    spacing: f32,
}

struct LevelTile {
    layer: u32,
    first: vec2<u32>,
}

struct TerrainInfo {
    origin: vec4<f32>,
    levels: vec4<u32>,
}

struct GroundHeight {
    found: bool,
    level: u32,
    tile: LevelTile,
    height: f32,
}

struct GroundTile {
    found: bool,
    level: u32,
    tile: LevelTile,
}

struct GroundMaterial {
    first: u32,
    second: u32,
    share: u32,
}

struct MaterialLook {
    far: vec4<f32>,
}

struct TileSummary {
    lowest: f32,
    highest: f32,
    materials: u32,
    site_low: vec4<f32>,
    site_high: vec4<f32>,
    wide_lowest: f32,
    wide_highest: f32,
    wide_materials: u32,
    wide_site_low: vec4<f32>,
    wide_site_high: vec4<f32>,
}

fn terrain_info(terrain: u32) -> TerrainInfo {
    var info: TerrainInfo;
    return info;
}

fn ground_of(info: TerrainInfo) -> GroundTerrain {
    var terrain: GroundTerrain;
    return terrain;
}

fn ground_extent(terrain: GroundTerrain) -> vec2<f32> {
    return vec2<f32>(0.0);
}

fn level_tiles(terrain: GroundTerrain, level: u32) -> vec2<u32> {
    return vec2<u32>(0u);
}

fn tile_layer(terrain: GroundTerrain, level: u32, tile: vec2<u32>) -> u32 {
    return NOT_RESIDENT;
}

fn level_tile(terrain: GroundTerrain, level: u32, samples: vec2<f32>) -> LevelTile {
    var tile: LevelTile;
    tile.layer = NOT_RESIDENT;
    return tile;
}

fn tile_height(tile: LevelTile, sample: vec2<u32>) -> f32 {
    return 0.0;
}

fn tile_peak(tile: LevelTile, texel: vec2<u32>, mip: u32) -> f32 {
    return 0.0;
}

fn tile_summary(tile: LevelTile, cell: vec2<u32>, mip: u32) -> TileSummary {
    var summary: TileSummary;
    return summary;
}

fn tile_gradient(tile: LevelTile, sample: vec2<u32>) -> vec2<f32> {
    return vec2<f32>(0.0);
}

fn tile_material(tile: LevelTile, sample: vec2<u32>) -> GroundMaterial {
    var material: GroundMaterial;
    return material;
}

fn triangle_height(a: f32, b: f32, c: f32, d: f32, f: vec2<f32>) -> f32 {
    return a;
}

fn true_height(terrain: GroundTerrain, local: vec2<f32>, level: u32, tile: LevelTile) -> f32 {
    return 0.0;
}

fn ground_tile(terrain: GroundTerrain, local: vec2<f32>, first: u32) -> GroundTile {
    var tile: GroundTile;
    return tile;
}

fn ground_height(terrain: GroundTerrain, local: vec2<f32>) -> GroundHeight {
    var height: GroundHeight;
    return height;
}

fn ground_colors_mip(terrain: GroundTerrain, footprint: f32) -> u32 {
    return 0u;
}

fn ground_colors(terrain: GroundTerrain, layer: u32, local: vec2<f32>, mip: u32) -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn macro_color(colors: vec4<f32>) -> vec3<f32> {
    return vec3<f32>(0.0);
}

fn ground_layer(layer: u32, uv: vec2<f32>, lod: f32) -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn material_look(base: u32, material: u32) -> MaterialLook {
    var look: MaterialLook;
    return look;
}

fn tile_site(tile: LevelTile, sample: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn site_shift(color: vec3<f32>, base: u32, material: u32, site: vec4<f32>) -> vec3<f32> {
    return color;
}

fn ground_sky(terrain: u32, local: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(1.0, 0.5, 0.5, 0.5);
}

fn detail_span(world: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn detail_resolved(span: vec2<f32>) -> f32 {
    return 0.0;
}
