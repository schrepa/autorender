// The original ranges are the first 2*N words of the arena. Full consumers
// keep their original vec2 binding and never need the optional tail.
fn depth_clusters() -> u32 {
    return clusters.counts.x * clusters.counts.y * clusters.counts.z;
}

fn depth_thresholds() -> u32 {
    return 5u * depth_clusters();
}

fn depth_header() -> u32 {
    return depth_thresholds() + 31u * clusters.counts.z;
}

fn depth_indices() -> u32 {
    return depth_header() + 8u;
}

fn depth_slice(cluster: u32) -> u32 {
    // CPU validation requires the existing 16 by 9 grid. A constant divisor
    // avoids backend-dependent lowering of run-time integer division.
    return cluster / 144u;
}

// The original forward cluster selection, shared with coverage collection.
fn light_cluster(depth: f32, pixel: vec2<f32>) -> u32 {
    let counts = clusters.counts;
    let tile = min(vec2<u32>(max(pixel, vec2<f32>(0.0)) / clusters.grid.xy),
        counts.xy - vec2<u32>(1u));
    let slice = u32(clamp(floor(log2(max(depth, 1.0e-6)) * clusters.grid.z + clusters.grid.w),
        0.0, f32(counts.z - 1u)));
    return (slice * counts.y + tile.y) * counts.x + tile.x;
}
