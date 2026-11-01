import package::face::ground_surface::{GroundMaterial, material_at};
import package::generated::terrain_shading::{CONTROL_GROUP, ControlJob, TILE_STRIDE};
import package::terrain_shading::tile::{tile_height, tile_shape};

@group(1) @binding(0) var<storage, read> control_jobs: array<ControlJob>;
@group(1) @binding(1) var control: texture_storage_2d_array<r32uint, write>;

@compute @workgroup_size(CONTROL_GROUP, CONTROL_GROUP)
fn tile_control(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= TILE_STRIDE || id.y >= TILE_STRIDE {
        return;
    }
    let job = control_jobs[id.z];
    let shape = tile_shape(id.x, id.y, job.spacing, job.layer);
    let at = vec2<f32>(f32(job.first.x + i32(id.x) - 1), f32(job.first.y + i32(id.y) - 1)) * job.spacing;
    let height = tile_height(id.x, id.y, job.layer);
    let material = material_at(job.terrain, at, height, length(shape.xy));
    let packed = (material.first & 0xffu) | ((material.second & 0xffu) << 8u)
        | ((material.share & 0xffu) << 16u) | ((material.wet & 0xffu) << 24u);
    textureStore(control, id.xy, job.layer, vec4<u32>(packed, 0u, 0u, 0u));
}
