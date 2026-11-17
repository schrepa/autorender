import package::engine::color::encode_srgb;

struct Post {
    size: vec4<f32>,
    kept: vec4<u32>,
}

@group(0) @binding(0) var<uniform> post: Post;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var scene: texture_2d<f32>;

fn uncovered(pixel: vec2<i32>) -> bool {
    return post.kept.x != 0u && textureLoad(depth_texture, pixel, 0) <= 0.0;
}

@fragment
fn fs_composite(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = min(vec2<i32>(position.xy), vec2<i32>(post.size.xy) - 1);
    if uncovered(pixel) {
        discard;
    }
    return vec4<f32>(encode_srgb(textureLoad(scene, pixel, 0).rgb), 1.0);
}
