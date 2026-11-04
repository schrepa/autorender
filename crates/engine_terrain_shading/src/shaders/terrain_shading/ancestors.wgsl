import package::generated::terrain_shading::{ANCESTORS_GROUP, AncestorJob, NOT_RESIDENT, TILE_STRIDE};

@group(1) @binding(0) var<storage, read> ancestor_jobs: array<AncestorJob>;
@group(1) @binding(1) var ancestor_texels: texture_storage_2d_array<r32uint, read_write>;

@compute @workgroup_size(ANCESTORS_GROUP, ANCESTORS_GROUP)
fn tile_ancestors(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= TILE_STRIDE || id.y >= TILE_STRIDE {
        return;
    }
    let job = ancestor_jobs[id.z];
    var value = 0u;
    if job.held_layer != NOT_RESIDENT {
        let half = vec2<i32>(1i << (job.shift - 1u));
        let sample = job.first + vec2<i32>(id.xy);
        let there = ((sample + half) >> vec2<u32>(job.shift)) - job.held_first;
        let texel = clamp(there, vec2<i32>(0), vec2<i32>(i32(TILE_STRIDE) - 1));
        value = textureLoad(ancestor_texels, texel, job.held_layer).x;
    }
    textureStore(ancestor_texels, id.xy, job.layer, vec4<u32>(value, 0u, 0u, 0u));
}
