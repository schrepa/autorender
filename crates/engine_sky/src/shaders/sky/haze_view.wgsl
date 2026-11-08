import package::engine::view::camera;

@group(1) @binding(1) var depth_texture: texture_depth_2d;

fn inverse_distance_at(depth: texture_depth_2d, pixel: vec2<i32>) -> f32 {
    let value = textureLoad(depth, pixel, 0);
    if value <= 0.0 {
        return 0.0;
    }
    let uv = (vec2<f32>(pixel) + 0.5) / camera.region.xy;
    let point = camera.inverse_view_projection
        * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, value, 1.0);
    return 1.0 / max(length(point.xyz / point.w - camera.position.xyz), 1.0e-4);
}

fn world_at(pixel: vec2<i32>) -> vec4<f32> {
    let depth = textureLoad(depth_texture, pixel, 0);
    let uv = (vec2<f32>(pixel) + 0.5) / camera.region.xy;
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = camera.inverse_view_projection * clip;
    return vec4<f32>(world.xyz / world.w, select(0.0, 1.0, depth > 0.0));
}
