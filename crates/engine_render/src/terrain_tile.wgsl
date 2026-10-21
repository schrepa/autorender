// A terrain tile's layer of the height array, as the passes deriving its
// maps from it read it. Modules listing this file bind the array as
// `heights` and a uniform `job` naming the tile's `layer`.

// Texels along a layer's side: a tile's samples and their border.
const TILE_STRIDE: u32 = 259u;

// The height at texel (`column`, `row`) of the tile's layer.
fn tile_height(column: u32, row: u32) -> f32 {
    return textureLoad(heights, vec2<u32>(column, row), job.layer, 0).r;
}

// The smooth surface through the tile's heights at texel (`column`, `row`),
// its samples `spacing` meters apart, from the 3 x 3 texels around it, the
// layer's edges clamped, as `engine_terrain::SampleShape` gives it: its
// gradient (rise per meter along x and z) and Laplacian (per meter).
fn tile_shape(column: u32, row: u32, spacing: f32) -> vec3<f32> {
    let last = TILE_STRIDE - 1u;
    let left = select(column - 1u, 0u, column == 0u);
    let right = min(column + 1u, last);
    let up = select(row - 1u, 0u, row == 0u);
    let down = min(row + 1u, last);
    let a = tile_height(left, up);
    let b = tile_height(column, up);
    let c = tile_height(right, up);
    let d = tile_height(left, row);
    let e = tile_height(column, row);
    let f = tile_height(right, row);
    let g = tile_height(left, down);
    let h = tile_height(column, down);
    let i = tile_height(right, down);
    // The spline's central differences, weighted 1, 4 and 1 across them.
    let across = 12.0 * spacing;
    return vec3<f32>(((c - a) + 4.0 * (f - d) + (i - g)) / across,
        ((g - a) + 4.0 * (h - b) + (i - c)) / across,
        (a + b + c + d + f + g + h + i - 8.0 * e) / (3.0 * spacing * spacing));
}
