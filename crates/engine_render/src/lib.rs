#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

pub use engine_core::RenderStats;

include!(concat!(env!("OUT_DIR"), "/source.rs"));

pub mod bindings;
pub mod cached;
pub mod gpu_calls;
mod gpu_data;
pub mod pipeline_layouts;

mod batches;
mod clouds;
pub mod composite;
mod debug_view;
mod depth_pyramid;
mod entries;
pub mod faces;
pub mod gpu_layouts;
mod gpu_uploads;
pub mod graph;
mod impostors;
pub mod lazy;
mod lights;
pub mod linker;
mod particles;
pub mod parts;
pub mod pieces;
mod pipelines;
mod probes;
pub mod render_bricks;
mod retained_uploads;
pub mod shaders;
mod shadow_fit;
mod sky;
mod uploads;

mod buffers;
mod environment;
mod exposure;
mod gpu_presentation;
mod ground_cover;
mod materials;
mod memory;
mod mesh_arena;
mod mipmaps;
mod occupancy;
mod platform;
mod post;
mod profiling;
mod renderer;
mod scatter;
mod scene_shading;
mod screen_effects;
mod shadows;
mod staging;
mod targets;
mod temporal;
mod terrain;
mod terrain_bounce;
mod terrain_horizons;
mod terrain_macro;
mod terrain_maps;
mod terrain_sky;
mod transparent;
mod water;
mod water_falls;
mod water_kernels;
mod water_particles;
mod water_settle;
mod water_sim;
mod world_trace;
pub use memory::GpuMemory;
pub use parts::{standard_features, standard_passes, standard_shared};
pub use platform::{AdapterInfo, RenderTarget, read_range};
#[cfg(not(target_arch = "wasm32"))]
pub use platform::{GpuPreference, NativeWindow, start_native_instances_with};
#[cfg(not(target_arch = "wasm32"))]
pub use renderer::GpuMark;
pub use renderer::{GpuPassTimes, Renderer, StartingRenderer};
