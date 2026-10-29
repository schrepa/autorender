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
pub mod gpu_data;
pub mod pipeline_layouts;

pub mod batches;
mod clouds;
pub mod composite;
pub mod debug_targets;
mod debug_view;
pub mod depth_pyramid;
pub mod entries;
pub mod faces;
pub mod gpu_layouts;
pub mod gpu_uploads;
pub mod graph;
pub mod impostors;
pub mod lazy;
mod lights;
pub mod linker;
mod particles;
pub mod parts;
pub mod pieces;
pub mod pipelines;
mod probes;
pub mod render_bricks;
pub mod retained_uploads;
pub mod shaders;
mod shadow_fit;
mod sky;
pub mod uploads;

pub mod buffers;
mod environment;
pub mod gpu_presentation;
mod ground_cover;
pub mod materials;
pub mod memory;
pub mod mesh_arena;
pub mod mipmaps;
mod occupancy;
pub mod platform;
pub mod post;
pub mod profiling;
pub mod renderer;
mod scatter;
pub mod scene_shading;
pub mod screen_effects;
mod shadows;
pub mod staging;
pub mod targets;
mod terrain;
mod terrain_bounce;
mod terrain_horizons;
mod terrain_macro;
mod terrain_maps;
mod terrain_sky;
pub mod transparent;
mod water;
mod water_falls;
mod water_kernels;
mod water_particles;
mod water_settle;
mod water_sim;
mod world_trace;
pub use memory::GpuMemory;
pub use platform::{AdapterInfo, RenderTarget, read_range};
#[cfg(not(target_arch = "wasm32"))]
pub use platform::{GpuPreference, NativeWindow, start_native_instances_with};
#[cfg(not(target_arch = "wasm32"))]
pub use renderer::GpuMark;
pub use renderer::{Core, GpuPassTimes, Renderer, StartingRenderer};
