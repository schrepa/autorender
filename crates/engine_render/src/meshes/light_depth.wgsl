// Only ordinary forward opaque variants enable the guard. Ineligible
// consumers compile out every interval and arena-tail read.
override DEPTH_OCCUPANCY: bool = false;

// Used only by the coverage entry point, with its own group layout. Never
// bound writable alongside the read-only ranges in the same pass.
@group(3) @binding(4) var<storage, read_write> light_depth_arena: array<atomic<u32>>;

fn depth_shading_word(index: u32) -> u32 {
    return cluster_ranges[index >> 1u][index & 1u];
}

fn depth_shading_bin(value: f32, slice: u32) -> u32 {
    let base = depth_thresholds() + slice * 31u;
    var low = 0u;
    var high = 31u;
    while low < high {
        let middle = (low + high) >> 1u;
        let threshold = bitcast<f32>(depth_shading_word(base + middle));
        if threshold <= value {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    return low;
}

fn depth_shading_guard(world: vec3<f32>, cluster: u32) -> bool {
    if clusters.lights.y == 0u || !depth_position_valid(world)
        || !depth_plane_valid(clusters.depth) {
        return false;
    }
    // A nonzero tag names one immutable view/phase/dimension/storage tuple.
    // Check the arena before any tail access; full allocations have no tail.
    if arrayLength(&cluster_ranges) < (depth_indices() + clusters.lights.z + 1u) / 2u {
        return false;
    }
    let generation = depth_shading_word(depth_header());
    if generation != clusters.lights.y
        || depth_shading_word(depth_header() + 1u) + 1u != clusters.lights.w {
        return false;
    }
    let occupied = depth_shading_word(4u * depth_clusters() + cluster);
    if occupied == 0u {
        return false;
    }
    let slice = depth_slice(cluster);
    let interval = depth_point_interval(world, clusters.depth);
    let fragment = depth_interval_bits(depth_shading_bin(interval.x, slice),
        depth_shading_bin(interval.y, slice));
    return depth_guard(occupied, fragment, clusters.lights.y, generation);
}

// All ordinary vertex variants put the displayed world position at location
// 1. Ignoring alpha/material discards collects a conservative superset. Depth
// always passes and is never written, so this also collects shaded overdraw.
@fragment
fn fs_light_depth(@builtin(position) pixel: vec4<f32>, @location(1) world_v: vec4<f32>) {
    let world = world_v.xyz;
    if !depth_position_valid(world) || !depth_plane_valid(clusters.depth) {
        return;
    }
    let depth = dot(clusters.depth.xyz, world) + clusters.depth.w;
    let cluster = light_cluster(depth, pixel.xy);
    let slice = depth_slice(cluster);
    let mask = depth_atomic_mask(depth_point_interval(world, clusters.depth), slice);
    atomicOr(&light_depth_arena[4u * depth_clusters() + cluster], mask);
}
