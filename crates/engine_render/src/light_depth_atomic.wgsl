// Collection and filtering read the same threshold table through the arena's
// atomic view, in passes separate from read-only shading.
fn depth_atomic_bin(value: f32, slice: u32) -> u32 {
    let base = depth_thresholds() + slice * 31u;
    var low = 0u;
    var high = 31u;
    while low < high {
        let middle = (low + high) >> 1u;
        let threshold = bitcast<f32>(atomicLoad(&light_depth_arena[base + middle]));
        if threshold <= value {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }
    return low;
}

fn depth_atomic_mask(interval: vec2<f32>, slice: u32) -> u32 {
    return depth_interval_bits(depth_atomic_bin(interval.x, slice),
        depth_atomic_bin(interval.y, slice));
}
