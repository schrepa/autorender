// Depth pyramids: level 0 holds the farthest depth of each 2x2 block of the
// depth buffer, every further level the farthest of each 2x2 block of the
// level below, for occlusion culling (decision 040); or the nearest depths,
// for screen-space reflections (decision 025). Depth is reversed (decision
// 042): the farthest is the smallest, the nearest the largest. Levels have
// power-of-two sizes, so each halves the last exactly; only the texels
// covering the depth buffer are written, and reads past the level below's
// repeat its edge texel, whose extreme already includes the buffer's edge.

// The written texels of the level written, and the last written texel of
// the level read (the depth buffer's last pixel for level zero).
struct Extent {
    size: vec2<u32>,
    last: vec2<u32>,
}

@group(0) @binding(0) var source_depth: texture_depth_2d;
@group(0) @binding(1) var source_level: texture_2d<f32>;
@group(0) @binding(2) var target_level: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> extent: Extent;

@compute @workgroup_size(8, 8)
fn pyramid_first(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= extent.size) {
        return;
    }
    let last = extent.last;
    let base = id.xy * 2u;
    let farthest = min(
        min(textureLoad(source_depth, min(base, last), 0),
            textureLoad(source_depth, min(base + vec2<u32>(1u, 0u), last), 0)),
        min(textureLoad(source_depth, min(base + vec2<u32>(0u, 1u), last), 0),
            textureLoad(source_depth, min(base + vec2<u32>(1u, 1u), last), 0)));
    textureStore(target_level, id.xy, vec4<f32>(farthest, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn pyramid_down(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= extent.size) {
        return;
    }
    let last = extent.last;
    let base = id.xy * 2u;
    let farthest = min(
        min(textureLoad(source_level, min(base, last), 0).r,
            textureLoad(source_level, min(base + vec2<u32>(1u, 0u), last), 0).r),
        min(textureLoad(source_level, min(base + vec2<u32>(0u, 1u), last), 0).r,
            textureLoad(source_level, min(base + vec2<u32>(1u, 1u), last), 0).r));
    textureStore(target_level, id.xy, vec4<f32>(farthest, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn pyramid_first_nearest(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= extent.size) {
        return;
    }
    let last = extent.last;
    let base = id.xy * 2u;
    let nearest = max(
        max(textureLoad(source_depth, min(base, last), 0),
            textureLoad(source_depth, min(base + vec2<u32>(1u, 0u), last), 0)),
        max(textureLoad(source_depth, min(base + vec2<u32>(0u, 1u), last), 0),
            textureLoad(source_depth, min(base + vec2<u32>(1u, 1u), last), 0)));
    textureStore(target_level, id.xy, vec4<f32>(nearest, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn pyramid_down_nearest(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= extent.size) {
        return;
    }
    let last = extent.last;
    let base = id.xy * 2u;
    let nearest = max(
        max(textureLoad(source_level, min(base, last), 0).r,
            textureLoad(source_level, min(base + vec2<u32>(1u, 0u), last), 0).r),
        max(textureLoad(source_level, min(base + vec2<u32>(0u, 1u), last), 0).r,
            textureLoad(source_level, min(base + vec2<u32>(1u, 1u), last), 0).r));
    textureStore(target_level, id.xy, vec4<f32>(nearest, 0.0, 0.0, 0.0));
}
