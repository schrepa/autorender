import package::generated::scatter::LAID_GROUP;

@group(1) @binding(0) var scene_depth: texture_depth_2d;
@group(1) @binding(1) var laid_out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(LAID_GROUP, LAID_GROUP)
fn copy_laid(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(laid_out)) || any(id.xy >= textureDimensions(scene_depth)) {
        return;
    }
    textureStore(laid_out, id.xy, vec4<f32>(textureLoad(scene_depth, id.xy, 0), 0.0, 0.0, 0.0));
}
