@group(0) @binding(0) var<uniform> clusters: Clusters;
@group(0) @binding(1) var<storage, read> lights: array<Light>;
@group(0) @binding(2) var<storage, read> full_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> light_depth_arena: array<atomic<u32>>;

@compute @workgroup_size(64)
fn reset_light_depth(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let count = depth_clusters();
    if index < count {
        atomicStore(&light_depth_arena[4u * count + index], 0u);
        atomicStore(&light_depth_arena[2u * count + 2u * index], 0u);
        atomicStore(&light_depth_arena[2u * count + 2u * index + 1u], 0u);
    }
    if index < 8u {
        var value = 0u;
        if index == 0u { value = clusters.lights.y; }
        if index == 1u { value = clusters.lights.w - 1u; }
        atomicStore(&light_depth_arena[depth_header() + index], value);
    }
}

var<workgroup> depth_prefix: array<u32, 64>;
var<workgroup> depth_full_range: vec2<u32>;
var<workgroup> depth_occupied: u32;

// Each cluster compacts only the entries the unmodified binner selected,
// into that same segment. Freed entries can never admit another light.
@compute @workgroup_size(64)
fn filter_light_depth(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let cluster = group.x;
    let count = depth_clusters();
    if lane == 0u {
        depth_full_range = vec2<u32>(atomicLoad(&light_depth_arena[2u * cluster]),
            atomicLoad(&light_depth_arena[2u * cluster + 1u]));
        depth_occupied = atomicLoad(&light_depth_arena[4u * count + cluster]);
    }
    let full = workgroupUniformLoad(&depth_full_range);
    let occupied = workgroupUniformLoad(&depth_occupied);
    let slice = depth_slice(cluster);
    let total = select(0u, full.y, occupied != 0u);
    var kept = 0u;
    for (var base = 0u; base < total; base += 64u) {
        let offset = base + lane;
        var index = 0u;
        var keep = 0u;
        if offset < total {
            index = full_indices[full.x + offset];
            let light = lights[index];
            keep = 1u;
            if depth_light_valid(light.position, light.range) {
                let interval = depth_light_interval(light.position, light.range, clusters.depth);
                keep = select(0u, 1u, (depth_atomic_mask(interval, slice) & occupied) != 0u);
            }
        }
        depth_prefix[lane] = keep;
        workgroupBarrier();
        for (var step = 1u; step < 64u; step <<= 1u) {
            var before = 0u;
            if lane >= step { before = depth_prefix[lane - step]; }
            workgroupBarrier();
            depth_prefix[lane] += before;
            workgroupBarrier();
        }
        if keep != 0u {
            let destination = depth_indices() + full.x + kept + depth_prefix[lane] - 1u;
            atomicStore(&light_depth_arena[destination], index);
        }
        kept += depth_prefix[63u];
        workgroupBarrier();
    }
    if lane == 0u {
        atomicStore(&light_depth_arena[2u * count + 2u * cluster], full.x);
        atomicStore(&light_depth_arena[2u * count + 2u * cluster + 1u], kept);
        let header = depth_header();
        if occupied == 0u {
            atomicAdd(&light_depth_arena[header + 7u], 1u);
        } else {
            atomicAdd(&light_depth_arena[header + 2u], 1u);
            atomicAdd(&light_depth_arena[header + 3u], countOneBits(occupied));
            atomicAdd(&light_depth_arena[header + 4u], full.y);
            atomicAdd(&light_depth_arena[header + 5u], kept);
            atomicAdd(&light_depth_arena[header + 6u], full.y - kept);
        }
    }
}
