import package::terrain_shading::tile::tile_shape;
import package::generated::terrain_shading::{NORMALS_GROUP, NormalJob, TILE_STRIDE};

@group(1) @binding(0) var<storage, read> normal_jobs: array<NormalJob>;
@group(1) @binding(1) var normals: texture_storage_2d_array<r32uint, write>;

@compute @workgroup_size(NORMALS_GROUP, NORMALS_GROUP)
fn tile_normals(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= TILE_STRIDE || id.y >= TILE_STRIDE {
        return;
    }
    let job = normal_jobs[id.z];
    let gradient = tile_shape(id.x, id.y, bitcast<f32>(job.spacing), job.layer).xy;
    textureStore(normals, id.xy, job.layer, vec4<u32>(pack2x16float(gradient), 0u, 0u, 0u));
}
