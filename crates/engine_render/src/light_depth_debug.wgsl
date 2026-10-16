// Requested development pages only. This kernel uses the production arena,
// original selected indices and final light placements of the named phase.
@group(1) @binding(0) var<uniform> depth_request: vec4<u32>;
@group(1) @binding(1) var<storage, read_write> depth_page: array<u32>;

@compute @workgroup_size(64)
fn inspect_light_depth(@builtin(local_invocation_index) lane: u32) {
    let cluster = depth_request.x;
    let offset = depth_request.y;
    let count = depth_clusters();
    let start = atomicLoad(&light_depth_arena[2u * cluster]);
    let length = atomicLoad(&light_depth_arena[2u * cluster + 1u]);
    let filtered_start = atomicLoad(&light_depth_arena[2u * count + 2u * cluster]);
    let filtered_length = atomicLoad(&light_depth_arena[2u * count + 2u * cluster + 1u]);
    if lane == 0u {
        depth_page[0] = atomicLoad(&light_depth_arena[depth_header()]);
        depth_page[1] = atomicLoad(&light_depth_arena[depth_header() + 1u]);
        depth_page[2] = cluster;
        depth_page[3] = atomicLoad(&light_depth_arena[4u * count + cluster]);
        depth_page[4] = start;
        depth_page[5] = length;
        depth_page[6] = filtered_start;
        depth_page[7] = filtered_length;
        depth_page[8] = offset;
        depth_page[9] = clusters.lights.z;
        depth_page[10] = count;
        depth_page[11] = 32u;
        depth_page[12] = atomicLoad(&light_depth_arena[depth_header() + 2u]);
        depth_page[13] = atomicLoad(&light_depth_arena[depth_header() + 4u]);
        depth_page[14] = atomicLoad(&light_depth_arena[depth_header() + 5u]);
        depth_page[15] = atomicLoad(&light_depth_arena[depth_header() + 6u]);
    }
    if lane < 31u {
        depth_page[16u + lane] = atomicLoad(&light_depth_arena[depth_thresholds() + depth_slice(cluster) * 31u + lane]);
    }
    if lane < 32u {
        let destination = 47u + lane * 6u;
        depth_page[destination] = 0xffffffffu;
        depth_page[destination + 1u] = 0xffffffffu;
        if offset < length && lane < length - offset {
            let index = full_indices[start + offset + lane];
            let light = lights[index];
            depth_page[destination] = index;
            depth_page[destination + 2u] = bitcast<u32>(light.position.x);
            depth_page[destination + 3u] = bitcast<u32>(light.position.y);
            depth_page[destination + 4u] = bitcast<u32>(light.position.z);
            depth_page[destination + 5u] = bitcast<u32>(light.range);
        }
        if offset < filtered_length && lane < filtered_length - offset {
            depth_page[destination + 1u] = atomicLoad(&light_depth_arena[depth_indices() + filtered_start + offset + lane]);
        }
    }
}
