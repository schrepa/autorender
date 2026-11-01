import package::engine::pyramid::{PYRAMID_GROUP, PyramidExtent, farthest_depth, farthest_level};

@group(1) @binding(0) var source_depth: texture_depth_2d;
@group(1) @binding(1) var source_level: texture_2d<f32>;
@group(1) @binding(2) var target_level: texture_storage_2d<r32float, write>;
@group(1) @binding(3) var<uniform> extent: PyramidExtent;

@compute @workgroup_size(PYRAMID_GROUP, PYRAMID_GROUP)
fn pyramid_first(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= extent.size) {
        return;
    }
    let farthest = farthest_depth(source_depth, id.xy * 2u, extent.last);
    textureStore(target_level, id.xy, vec4<f32>(farthest, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(PYRAMID_GROUP, PYRAMID_GROUP)
fn pyramid_down(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= extent.size) {
        return;
    }
    let farthest = farthest_level(source_level, id.xy * 2u, extent.last);
    textureStore(target_level, id.xy, vec4<f32>(farthest, 0.0, 0.0, 0.0));
}
