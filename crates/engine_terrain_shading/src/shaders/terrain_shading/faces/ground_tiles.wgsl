import package::engine::view::{camera, detail_across};
import package::place::ground_tiles::{GroundHeight, GroundMaterial, GroundTerrain, GroundTile,
    LevelTile, MaterialLook, NOT_RESIDENT, PALETTE_ENTRIES, PEAK_CELLS, PEAK_MIPS, TILE_QUADS,
    TerrainInfo, TileSummary};
import package::generated::terrain_shading::{CONTROL_PLANE, DETAIL_ANISOTROPY, NORMAL_PLANE,
    PaletteEntry, SITE_PLANE, SUMMARY_FIRST, SUMMARY_LAST, SUMMARY_PLANES, TABLE_LOOK, TILE_PLANES,
    control_a, control_b, control_blend};
import package::terrain_shading::draw_shared::{terrain_site_gain, terrain_site_lush};
import package::terrain_shading::tables::{table_look, table_terrain, terrain_tables};

@group(0) @binding(0) var heights: texture_2d_array<f32>;
@group(0) @binding(2) var tile_maps: texture_2d_array<u32>;
@group(0) @binding(4) var macro_maps: texture_2d_array<f32>;
@group(0) @binding(5) var<uniform> palette: array<PaletteEntry, PALETTE_ENTRIES>;
@group(0) @binding(7) var ground_sky_map: texture_2d_array<f32>;
@group(0) @binding(8) var ground_sky_sampler: sampler;
@group(0) @binding(9) var tile_summaries: texture_2d_array<u32>;

fn terrain_info(terrain: u32) -> TerrainInfo {
    let held = table_terrain(terrain);
    var info: TerrainInfo;
    info.origin = vec4<f32>(held.origin.xyz, held.anchor.w);
    info.levels = vec4<u32>(held.directory.x, held.directory.y, held.directory.z, held.macro_map.x);
    return info;
}

fn ground_sky(terrain: u32, local: vec2<f32>) -> vec4<f32> {
    let layer = table_terrain(terrain).directory.w;
    if layer == 0u {
        return vec4<f32>(1.0, 0.5, 0.5, 0.5);
    }
    let extent = ground_extent(ground_of(terrain_info(terrain)));
    return textureSampleLevel(ground_sky_map, ground_sky_sampler, local / extent, layer - 1u, 0.0);
}

fn ground_of(info: TerrainInfo) -> GroundTerrain {
    return GroundTerrain(info.levels.x, info.levels.y, info.origin.w);
}

fn ground_extent(terrain: GroundTerrain) -> vec2<f32> {
    return vec2<f32>(f32(terrain_tables[terrain.directory + 1u]),
        f32(terrain_tables[terrain.directory + 2u])) * 256.0 * terrain.spacing;
}

fn level_tiles(terrain: GroundTerrain, level: u32) -> vec2<u32> {
    let header = terrain.directory + level * 4u;
    return vec2<u32>(terrain_tables[header + 1u], terrain_tables[header + 2u]);
}

fn tile_layer(terrain: GroundTerrain, level: u32, tile: vec2<u32>) -> u32 {
    let header = terrain.directory + level * 4u;
    let tiles = vec2<u32>(terrain_tables[header + 1u], terrain_tables[header + 2u]);
    return terrain_tables[terrain_tables[header] + tile.y * tiles.x + tile.x];
}

fn level_tile(terrain: GroundTerrain, level: u32, samples: vec2<f32>) -> LevelTile {
    let header = terrain.directory + level * 4u;
    let tiles = vec2<u32>(terrain_tables[header + 1u], terrain_tables[header + 2u]);
    let tile = min(vec2<u32>(max(samples, vec2<f32>(0.0)) / 256.0), tiles - vec2<u32>(1u));
    var found: LevelTile;
    found.layer = terrain_tables[terrain_tables[header] + tile.y * tiles.x + tile.x];
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

fn tile_summary(tile: LevelTile, cell: vec2<u32>, mip: u32) -> TileSummary {
    let level = clamp(mip, SUMMARY_FIRST, SUMMARY_LAST);
    let at = min(cell >> vec2<u32>(level - min(mip, level)),
        vec2<u32>((TILE_QUADS >> level) - 1u));
    let plane = tile.layer * SUMMARY_PLANES;
    let wide = textureLoad(tile_summaries, at, plane, level - SUMMARY_FIRST);
    let sites = textureLoad(tile_summaries, at, plane + 1u, level - SUMMARY_FIRST);
    let own = textureLoad(tile_summaries, at, plane + 2u, level - SUMMARY_FIRST);
    return TileSummary(bitcast<f32>(own.x), bitcast<f32>(own.y), wide.w,
        unpack4x8unorm(sites.x), unpack4x8unorm(sites.y), bitcast<f32>(wide.x),
        bitcast<f32>(wide.y), wide.z, unpack4x8unorm(sites.z), unpack4x8unorm(sites.w));
}

fn tile_gradient(tile: LevelTile, sample: vec2<u32>) -> vec2<f32> {
    let texel = min(sample - tile.first, vec2<u32>(256u)) + vec2<u32>(1u);
    return unpack2x16float(textureLoad(tile_maps, texel, tile.layer * TILE_PLANES + NORMAL_PLANE,
        0).x);
}

fn tile_material(tile: LevelTile, sample: vec2<u32>) -> GroundMaterial {
    let texel = min(sample - tile.first, vec2<u32>(256u)) + vec2<u32>(1u);
    let value = textureLoad(tile_maps, texel, tile.layer * TILE_PLANES + CONTROL_PLANE, 0).x;
    return GroundMaterial(control_a(value), control_b(value), control_blend(value));
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

fn ground_layer(layer: u32, uv: vec2<f32>, lod: f32) -> vec4<f32> {
    if layer == 0u {
        return vec4<f32>(0.0);
    }
    return textureSampleLevel(macro_maps, ground_sky_sampler, uv, layer - 1u, lod);
}

fn material_look(base: u32, material: u32) -> MaterialLook {
    let entry = palette[min(base + material, PALETTE_ENTRIES - 1u)];
    return MaterialLook(entry.far);
}

fn tile_site(tile: LevelTile, sample: vec2<u32>) -> vec4<f32> {
    let texel = min(sample - tile.first, vec2<u32>(256u)) + vec2<u32>(1u);
    return unpack4x8unorm(textureLoad(tile_maps, texel, tile.layer * TILE_PLANES + SITE_PLANE,
        0).x);
}

fn site_shift(color: vec3<f32>, base: u32, material: u32, site: vec4<f32>) -> vec3<f32> {
    let entry = palette[min(base + material, PALETTE_ENTRIES - 1u)];
    let look = table_look(TABLE_LOOK);
    return color * terrain_site_gain(entry.heights.y, terrain_site_lush(site), look.lush.rgb,
        look.bare.rgb);
}

const DETAIL_EDGE_ON: f32 = 0.0625;

fn detail_span(world: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
    let across = detail_across(world);
    let toward = world - camera.detail.xyz;
    let ray = toward / max(length(toward), 1.0e-6);
    let cosine = dot(normal, ray);
    let facing = max(abs(cosine), DETAIL_EDGE_ON);
    let laid = ray - normal * cosine;
    let way = laid.xz * inverseSqrt(max(dot(laid, laid), 1.0e-12));
    return vec4<f32>(across, across / facing, way);
}

fn detail_resolved(span: vec2<f32>) -> f32 {
    return max(min(span.x, span.y), max(span.x, span.y) / DETAIL_ANISOTROPY);
}
