import package::terrain::tile::TILE_STRIDE;

struct Job {
    layer: u32,
    spacing: u32,
    pad: vec2<u32>,
}

@group(0) @binding(0) var heights: texture_2d_array<f32>;
@group(0) @binding(1) var<uniform> job: Job;
@group(0) @binding(2) var normals: texture_storage_2d<r32uint, write>;

fn tile_height(column: u32, row: u32) -> f32 {
    return textureLoad(heights, vec2<u32>(column, row), job.layer, 0).r;
}

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
    let across = 12.0 * spacing;
    return vec3<f32>(((c - a) + 4.0 * (f - d) + (i - g)) / across,
        ((g - a) + 4.0 * (h - b) + (i - c)) / across,
        (a + b + c + d + f + g + h + i - 8.0 * e) / (3.0 * spacing * spacing));
}

@compute @workgroup_size(16, 16)
fn tile_normals(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= TILE_STRIDE || id.y >= TILE_STRIDE {
        return;
    }
    let gradient = tile_shape(id.x, id.y, bitcast<f32>(job.spacing)).xy;
    textureStore(normals, id.xy, vec4<u32>(pack2x16float(gradient), 0u, 0u, 0u));
}
