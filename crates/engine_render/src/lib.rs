//! WebGPU rendering of engine frames, without access to the gameplay world.
//!
//! The browser renderer owns one canvas and all its GPU resources. CPU upload
//! records and WGSL live together; ordinary gameplay never handles GPU objects.

pub use engine_core::RenderStats;

pub mod bindings;
mod cached;
mod gpu_data;

mod batches;
mod clouds;
mod decals;
mod deformation;
mod depth_pyramid;
mod entries;
pub mod gpu_layouts;
mod gpu_uploads;
mod impostors;
mod lights;
mod particles;
mod pipelines;
mod probes;
mod render_graph;
mod retained_uploads;
pub mod shaders;
mod shadow_fit;
mod sky;
mod uploads;
mod virtual_streaming;

mod browser;
mod buffers;
mod environment;
mod gpu_presentation;
mod ground_cover;
mod materials;
mod mesh_arena;
mod mipmaps;
mod occupancy;
mod platform;
mod post;
mod profiling;
mod scatter;
mod shadows;
mod staging;
mod targets;
mod temporal;
mod terrain;
mod terrain_horizons;
mod terrain_maps;
mod terrain_sky;
mod terrain_vt;
mod transparent;
mod virtual_geometry;
mod visibility;
mod water;
pub use browser::{Renderer, StartingRenderer};
#[cfg(not(target_arch = "wasm32"))]
pub use platform::NativeWindow;
pub use platform::{RenderError, RenderTarget, read_range};
