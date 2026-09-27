// Depth pyramid of occlusion culling (decision 040): level 0 holds the
// farthest depth of each 2x2 block of the depth buffer, every further level
// the farthest of each 2x2 block of the level below. Levels have power-of-two
// sizes, so each halves the last exactly; texels beyond the depth buffer
// repeat its edge, which cannot lower a maximum.

@group(0) @binding(0) var source_depth: texture_depth_2d;
@group(0) @binding(1) var source_level: texture_2d<f32>;
@group(0) @binding(2) var target_level: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn pyramid_first(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(target_level)) {
        return;
    }
    let last = textureDimensions(source_depth) - 1u;
    let base = id.xy * 2u;
    let farthest = max(
        max(textureLoad(source_depth, min(base, last), 0),
            textureLoad(source_depth, min(base + vec2<u32>(1u, 0u), last), 0)),
        max(textureLoad(source_depth, min(base + vec2<u32>(0u, 1u), last), 0),
            textureLoad(source_depth, min(base + vec2<u32>(1u, 1u), last), 0)));
    textureStore(target_level, id.xy, vec4<f32>(farthest, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn pyramid_down(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(target_level)) {
        return;
    }
    let last = textureDimensions(source_level) - 1u;
    let base = id.xy * 2u;
    let farthest = max(
        max(textureLoad(source_level, min(base, last), 0).r,
            textureLoad(source_level, min(base + vec2<u32>(1u, 0u), last), 0).r),
        max(textureLoad(source_level, min(base + vec2<u32>(0u, 1u), last), 0).r,
            textureLoad(source_level, min(base + vec2<u32>(1u, 1u), last), 0).r));
    textureStore(target_level, id.xy, vec4<f32>(farthest, 0.0, 0.0, 0.0));
}
