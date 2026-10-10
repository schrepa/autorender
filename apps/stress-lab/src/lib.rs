//! Deterministic stress workloads built exclusively through the public engine SDK.
mod authored;
#[cfg(target_arch = "wasm32")]
mod browser;
mod catalog;
mod caves;
mod city;
mod config;
mod dense;
mod highlands;
mod metrics;
mod metro;
mod observation;
mod rocks;
mod scene;
mod shapes;
mod trees;

pub use authored::{AuthoredScene, PlayerWalk};
pub use catalog::{Catalog, Choice, catalog};
pub use config::{MESH_VARIETIES, StressConfig, StressScene};
pub use dense::FILES as DENSE_FILES;
pub use highlands::{FILES as HIGHLANDS_FILES, ROCK_FILES};
pub use metrics::{FrameMetrics, StressMetrics};
pub use observation::{StressSample, StressSnapshot, stress_observation, stress_sample};
pub use scene::{
    StressFile, create_workload, create_workload_with, look, set_view, stress_reflections,
    stress_shafts,
};

// Applications opt into allocator accounting explicitly; the engine never
// replaces a consumer's allocator. Normal builds retain Rust's default allocator.
#[cfg(feature = "profiling")]
#[global_allocator]
static ALLOCATOR: engine::profiling::TrackingAllocator<std::alloc::System> =
    engine::profiling::TrackingAllocator::new(std::alloc::System);
