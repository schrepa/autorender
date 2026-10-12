// A terrain tile's layer of the height array, as the passes deriving its
// maps from it read it. Modules listing this file bind the array as
// `heights` and a uniform `job` naming the tile's `layer`.

// Texels along a layer's side: a tile's samples and their border.
const TILE_STRIDE: u32 = 259u;

// The height at texel (`column`, `row`) of the tile's layer.
fn tile_height(column: u32, row: u32) -> f32 {
    return textureLoad(heights, vec2<u32>(column, row), job.layer, 0).r;
}
