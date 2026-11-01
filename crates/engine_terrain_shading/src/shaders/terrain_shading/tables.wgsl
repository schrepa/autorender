import package::place::heightfield::{TerrainUniform, TerrainView};
import package::generated::terrain_shading::{TABLE_HOLES, TABLE_TERRAINS, TABLE_VIEWS, TERRAIN_WORDS, VIEW_WORDS};

@group(0) @binding(0) var<storage, read> terrain_tables: array<u32>;

fn table_words(at: u32) -> vec4<u32> {
    return vec4<u32>(terrain_tables[at], terrain_tables[at + 1u], terrain_tables[at + 2u],
        terrain_tables[at + 3u]);
}

fn table_floats(at: u32) -> vec4<f32> {
    return bitcast<vec4<f32>>(table_words(at));
}

fn table_terrain(index: u32) -> TerrainUniform {
    let at = terrain_tables[TABLE_TERRAINS] + index * TERRAIN_WORDS;
    return TerrainUniform(table_floats(at), table_floats(at + 4u), table_words(at + 8u),
        table_floats(at + 12u), table_words(at + 16u), table_floats(at + 20u),
        table_floats(at + 24u), table_words(at + 28u), table_floats(at + 32u),
        table_floats(at + 36u), table_words(at + 40u));
}

fn table_view(index: u32) -> TerrainView {
    let at = terrain_tables[TABLE_VIEWS] + index * VIEW_WORDS;
    return TerrainView(
        mat4x4<f32>(table_floats(at), table_floats(at + 4u), table_floats(at + 8u),
            table_floats(at + 12u)),
        table_floats(at + 16u),
        mat4x4<f32>(table_floats(at + 20u), table_floats(at + 24u), table_floats(at + 28u),
            table_floats(at + 32u)));
}

fn table_fades(layer: u32) -> u32 {
    return terrain_tables[terrain_tables[TABLE_HOLES] + layer / 2u];
}
