import package::terrain::tile::{TILE_STRIDE, tile_shape};

struct Job {
    layer: u32,
    spacing: u32,
    pad: vec2<u32>,
}

@group(0) @binding(1) var<uniform> job: Job;
@group(0) @binding(2) var normals: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(16, 16)
fn tile_normals(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= TILE_STRIDE || id.y >= TILE_STRIDE {
        return;
    }
    let gradient = tile_shape(id.x, id.y, bitcast<f32>(job.spacing), job.layer).xy;
    textureStore(normals, id.xy, vec4<u32>(pack2x16float(gradient), 0u, 0u, 0u));
}
