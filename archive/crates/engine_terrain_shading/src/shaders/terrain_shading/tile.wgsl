import package::generated::terrain_shading::TILE_STRIDE;
import package::terrain_shading::field::terrain_heights;


fn tile_height(column: u32, row: u32, layer: u32) -> f32 {
    return textureLoad(terrain_heights, vec2<u32>(column, row), layer, 0).r;
}

fn tile_shape(column: u32, row: u32, spacing: f32, layer: u32) -> vec3<f32> {
    let last = TILE_STRIDE - 1u;
    let left = select(column - 1u, 0u, column == 0u);
    let right = min(column + 1u, last);
    let up = select(row - 1u, 0u, row == 0u);
    let down = min(row + 1u, last);
    let a = tile_height(left, up, layer);
    let b = tile_height(column, up, layer);
    let c = tile_height(right, up, layer);
    let d = tile_height(left, row, layer);
    let e = tile_height(column, row, layer);
    let f = tile_height(right, row, layer);
    let g = tile_height(left, down, layer);
    let h = tile_height(column, down, layer);
    let i = tile_height(right, down, layer);
    let across = 12.0 * spacing;
    return vec3<f32>(((c - a) + 4.0 * (f - d) + (i - g)) / across,
        ((g - a) + 4.0 * (h - b) + (i - c)) / across,
        (a + b + c + d + f + g + h + i - 8.0 * e) / (3.0 * spacing * spacing));
}
