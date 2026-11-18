struct Post {
    size: vec4<f32>,
    kept: vec4<u32>,
}

@group(0) @binding(0) var<uniform> post: Post;
