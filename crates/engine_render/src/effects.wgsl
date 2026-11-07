struct Post {
    inverse_view_projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    eye: vec4<f32>,
    size: vec4<f32>,
    extent: vec4<f32>,
    detail: vec4<f32>,
    kept: vec4<f32>,
    applied: vec4<u32>,
}

@group(0) @binding(0) var<uniform> post: Post;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var scene: texture_2d<f32>;
@group(0) @binding(7) var linear: sampler;

fn drawn_uv(pixel: vec2<f32>, settings: Post) -> vec2<f32> {
    return clamp(pixel, vec2<f32>(0.5), settings.size.xy - 0.5) * settings.extent.zw;
}

fn drawn_texels(size: vec2<f32>, settings: Post) -> vec2<f32> {
    let share = max(floor(size * settings.size.xy * settings.extent.zw), vec2<f32>(1.0));
    return select(share, size, settings.size.xy == settings.extent.xy);
}
