fn farthest_depth(source: texture_depth_2d, base: vec2<u32>, last: vec2<u32>) -> f32 {
    return min(
        min(textureLoad(source, min(base, last), 0),
            textureLoad(source, min(base + vec2<u32>(1u, 0u), last), 0)),
        min(textureLoad(source, min(base + vec2<u32>(0u, 1u), last), 0),
            textureLoad(source, min(base + vec2<u32>(1u, 1u), last), 0)));
}

fn farthest_level(source: texture_2d<f32>, base: vec2<u32>, last: vec2<u32>) -> f32 {
    return min(
        min(textureLoad(source, min(base, last), 0).r,
            textureLoad(source, min(base + vec2<u32>(1u, 0u), last), 0).r),
        min(textureLoad(source, min(base + vec2<u32>(0u, 1u), last), 0).r,
            textureLoad(source, min(base + vec2<u32>(1u, 1u), last), 0).r));
}

fn nearest_of_depth(source: texture_depth_2d, base: vec2<u32>, last: vec2<u32>) -> f32 {
    return max(
        max(textureLoad(source, min(base, last), 0),
            textureLoad(source, min(base + vec2<u32>(1u, 0u), last), 0)),
        max(textureLoad(source, min(base + vec2<u32>(0u, 1u), last), 0),
            textureLoad(source, min(base + vec2<u32>(1u, 1u), last), 0)));
}

fn nearest_of_level(source: texture_2d<f32>, base: vec2<u32>, last: vec2<u32>) -> f32 {
    return max(
        max(textureLoad(source, min(base, last), 0).r,
            textureLoad(source, min(base + vec2<u32>(1u, 0u), last), 0).r),
        max(textureLoad(source, min(base + vec2<u32>(0u, 1u), last), 0).r,
            textureLoad(source, min(base + vec2<u32>(1u, 1u), last), 0).r));
}
