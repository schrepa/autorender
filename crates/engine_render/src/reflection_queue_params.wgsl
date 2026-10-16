// One immutable, dynamically selected record per dispatch.
struct QueueParams {
    size: vec4<u32>, // Half width, height, tile width, tile height.
    scan: vec4<u32>, // Level offset, length, parent offset, dispatch width.
    extra: vec4<u32>, // Bin, full pixel capacity, dispatch row log2, reserved.
}
