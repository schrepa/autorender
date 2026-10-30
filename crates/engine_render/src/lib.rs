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
pub mod composite;
pub mod debug_targets;
pub mod depth_pyramid;
pub mod entries;
pub mod faces;
pub mod generated;
pub mod gpu_layouts;
pub mod gpu_uploads;
pub mod graph;
pub mod impostors;
pub mod lazy;
pub mod linker;
pub mod parts;
pub mod pieces;
pub mod pipelines;
pub mod render_bricks;
pub mod retained_uploads;
pub mod shaders;
pub mod uploads;

pub mod buffers;
pub mod gpu_presentation;
pub mod materials;
pub mod memory;
pub mod mesh_arena;
pub mod mipmaps;
pub mod platform;
pub mod post;
pub mod profiling;
pub mod renderer;
pub mod scene_shading;
pub mod screen_effects;
pub mod staging;
pub mod targets;
pub mod transparent;
pub use memory::GpuMemory;
pub use platform::{AdapterInfo, RenderTarget, read_range};
#[cfg(not(target_arch = "wasm32"))]
pub use platform::{GpuPreference, NativeWindow, start_native_instances_with};
#[cfg(not(target_arch = "wasm32"))]
pub use renderer::GpuMark;
pub use renderer::{Core, GpuPassTimes, Renderer, StartingRenderer};
