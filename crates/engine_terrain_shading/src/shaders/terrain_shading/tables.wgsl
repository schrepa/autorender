import package::place::heightfield::{TerrainUniform, TerrainView};
import package::generated::terrain_shading::{LookUniform, TABLE_FADES, TABLE_NOW, TABLE_TERRAINS, TABLE_VIEWS, TERRAIN_WORDS, VIEW_WORDS};

@group(0) @binding(0) var<storage, read> terrain_tables: array<u32>;

fn table_words(at: u32) -> vec4<u32> {
    return vec4<u32>(terrain_tables[at], terrain_tables[at + 1u], terrain_tables[at + 2u],
        terrain_tables[at + 3u]);
}

fn table_floats(at: u32) -> vec4<f32> {
    return bitcast<vec4<f32>>(table_words(at));
}

const TERRAIN_ORIGIN: u32 = 12u;
const TERRAIN_DIRECTORY: u32 = 16u;
const TERRAIN_MACRO_MAP: u32 = 24u;

fn terrain_at(index: u32) -> u32 {
    return terrain_tables[TABLE_TERRAINS] + index * TERRAIN_WORDS;
}

fn table_terrain(index: u32) -> TerrainUniform {
    let at = terrain_at(index);
    return TerrainUniform(table_floats(at), table_floats(at + 4u), table_words(at + 8u),
        table_floats(at + TERRAIN_ORIGIN), table_words(at + TERRAIN_DIRECTORY),
        table_floats(at + 20u), table_words(at + TERRAIN_MACRO_MAP), table_floats(at + 28u),
        table_floats(at + 32u), table_words(at + 36u));
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

fn table_look(at: u32) -> LookUniform {
    return LookUniform(table_floats(at), table_floats(at + 4u), table_floats(at + 8u),
        table_floats(at + 12u), table_floats(at + 16u), table_floats(at + 20u),
        table_floats(at + 24u), table_floats(at + 28u), table_floats(at + 32u),
        table_floats(at + 36u), table_floats(at + 40u), table_floats(at + 44u));
}

fn table_fades(layer: u32, seconds: f32) -> vec2<f32> {
    let at = terrain_tables[TABLE_FADES] + 2u * layer;
    let began = vec2<f32>(bitcast<f32>(terrain_tables[at]), bitcast<f32>(terrain_tables[at + 1u]));
    let since = max(vec2<f32>(bitcast<f32>(terrain_tables[TABLE_NOW])) - began, vec2<f32>(0.0));
    return vec2<f32>(1.0) - min(since / seconds, vec2<f32>(1.0));
}
