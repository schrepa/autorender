struct Post {
    size: vec4<f32>,
    kept: vec4<u32>,
}

@group(0) @binding(0) var<uniform> post: Post;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var scene: texture_2d<f32>;
