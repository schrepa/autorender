import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::place::heightfield::TerrainUniform;
import package::terrain_shading::tables::{table_terrain, terrain_tables};

fn terrain_holed(ground: TerrainUniform, cell: vec2<f32>) -> bool {
    let base = ground.directory.x;
    let levels = ground.directory.y;
    for (var index = 0u; index < levels; index++) {
        let header = base + index * 4u;
        let samples = max(cell / f32(1u << index), vec2<f32>(0.0));
        let tiles = vec2<u32>(terrain_tables[header + 1u], terrain_tables[header + 2u]);
        let tile = min(vec2<u32>(samples / 256.0), tiles - vec2<u32>(1u));
        let layer = terrain_tables[terrain_tables[header] + tile.y * tiles.x + tile.x];
        if layer == 0xffffffffu {
            continue;
        }
        let holes = terrain_tables[ground.surface.y + layer];
        if holes == 0xffffffffu {
            return false;
        }
        let at = vec2<u32>(clamp(floor(samples - vec2<f32>(tile * 256u)), vec2<f32>(0.0),
            vec2<f32>(256.0)));
        let bit = at.y * 257u + at.x;
        return (terrain_tables[holes + bit / 32u] & (1u << (bit % 32u))) != 0u;
    }
    return false;
}

fn opacity(input: SurfaceInput, point: SurfacePoint) -> f32 {
    if input.surface.x > 0.0 {
        return 1.0;
    }
    return select(1.0, 0.0, terrain_holed(table_terrain(input.object), input.uv));
}
