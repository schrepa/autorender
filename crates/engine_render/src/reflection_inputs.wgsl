// Immutable reflection inputs, shared by fragment and compute stages.
@group(0) @binding(0) var<uniform> post: Post;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
// Linear color and the ambient share, sampled from the sRGB scene target.
@group(0) @binding(2) var scene: texture_2d<f32>;
// Exposed glow weight x / (1 + x), and reflectivity.
@group(0) @binding(3) var glow: texture_2d<f32>;
@group(0) @binding(7) var linear: sampler;
// Nearest reversed depths of successive 2x2 blocks.
@group(0) @binding(10) var nearest_depth: texture_2d<f32>;
// Octahedral shading normal in [0, 1], and roughness.
@group(0) @binding(13) var surface_texture: texture_2d<f32>;
// Blurred scene levels for rough surfaces.
@group(0) @binding(14) var scene_levels: texture_2d<f32>;
