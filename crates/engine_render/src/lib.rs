//! WebGPU rendering of engine frames, without access to the gameplay world.
//!
//! The browser renderer owns one canvas and all its GPU resources. CPU upload
//! records and WGSL live together; ordinary gameplay never handles GPU objects.

pub use engine_core::RenderStats;

pub mod bindings;
#[cfg(target_arch = "wasm32")]
mod cached;
#[cfg(target_arch = "wasm32")]
mod gpu_data;

#[cfg(target_arch = "wasm32")]
mod batches;
#[cfg(target_arch = "wasm32")]
mod clouds;
#[cfg(target_arch = "wasm32")]
mod decals;
#[cfg(target_arch = "wasm32")]
mod deformation;
#[cfg(target_arch = "wasm32")]
mod depth_pyramid;
#[cfg(target_arch = "wasm32")]
mod entries;
pub mod gpu_layouts;
#[cfg(target_arch = "wasm32")]
mod gpu_uploads;
#[cfg(target_arch = "wasm32")]
mod impostors;
#[cfg(target_arch = "wasm32")]
mod lights;
#[cfg(target_arch = "wasm32")]
mod particles;
#[cfg(target_arch = "wasm32")]
mod pipelines;
#[cfg(target_arch = "wasm32")]
mod probes;
mod render_graph;
#[cfg(target_arch = "wasm32")]
mod retained_uploads;
pub mod shaders;
#[cfg(target_arch = "wasm32")]
mod shadow_fit;
#[cfg(target_arch = "wasm32")]
mod sky;
#[cfg(target_arch = "wasm32")]
mod uploads;
#[cfg(target_arch = "wasm32")]
mod virtual_streaming;

#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(target_arch = "wasm32")]
mod buffers;
#[cfg(target_arch = "wasm32")]
mod environment;
#[cfg(target_arch = "wasm32")]
mod gpu_presentation;
#[cfg(target_arch = "wasm32")]
mod ground_cover;
#[cfg(target_arch = "wasm32")]
mod materials;
#[cfg(target_arch = "wasm32")]
mod mesh_arena;
#[cfg(target_arch = "wasm32")]
mod mipmaps;
#[cfg(target_arch = "wasm32")]
mod occupancy;
#[cfg(target_arch = "wasm32")]
mod post;
#[cfg(target_arch = "wasm32")]
mod profiling;
#[cfg(target_arch = "wasm32")]
mod scatter;
#[cfg(target_arch = "wasm32")]
mod shadows;
#[cfg(target_arch = "wasm32")]
mod staging;
#[cfg(target_arch = "wasm32")]
mod targets;
#[cfg(target_arch = "wasm32")]
mod temporal;
#[cfg(target_arch = "wasm32")]
mod terrain;
#[cfg(target_arch = "wasm32")]
mod terrain_horizons;
#[cfg(target_arch = "wasm32")]
mod terrain_maps;
#[cfg(target_arch = "wasm32")]
mod terrain_sky;
#[cfg(target_arch = "wasm32")]
mod terrain_vt;
#[cfg(target_arch = "wasm32")]
mod transparent;
#[cfg(target_arch = "wasm32")]
mod virtual_geometry;
#[cfg(target_arch = "wasm32")]
mod visibility;
#[cfg(target_arch = "wasm32")]
mod water;
#[cfg(target_arch = "wasm32")]
pub use browser::{Renderer, StartingRenderer};
