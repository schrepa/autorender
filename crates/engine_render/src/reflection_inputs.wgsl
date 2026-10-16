// Immutable reflection inputs, shared by fragment and compute stages.
@group(0) @binding(0) var<uniform> post: Post;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var scene: texture_2d<f32>;
@group(0) @binding(3) var glow: texture_2d<f32>;
@group(0) @binding(7) var linear: sampler;
@group(0) @binding(10) var nearest_depth: texture_2d<f32>;
@group(0) @binding(13) var surface_texture: texture_2d<f32>;
@group(0) @binding(14) var scene_levels: texture_2d<f32>;
