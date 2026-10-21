//! Deterministic stress workloads built exclusively through the public engine SDK.

mod authored;
#[cfg(target_arch = "wasm32")]
mod browser;
mod catalog;
mod city;
mod config;
mod dense;
mod field;
mod game;
mod highlands;
mod kit;
mod metrics;
mod metro;
#[cfg(not(target_arch = "wasm32"))]
pub mod native;
mod observation;
mod presets;
mod scene;
mod world;

// `SOURCE`, the hash of the Stress Lab's code and the engine it builds on,
// which keys the records the worlds' own code makes (their generated meshes
// and textures), which hosts keep and the cook ships, and which the scenes'
// content names (`engine::Game::content`): their pipeline lists last while
// this code stays the same.
include!(concat!(env!("OUT_DIR"), "/source.rs"));

pub use authored::{AuthoredScene, PlayerWalk};
pub use catalog::{Catalog, Choice, Switch, catalog, switches};
pub use config::{MESH_VARIETIES, StressConfig, StressScene};
pub use dense::FILES as DENSE_FILES;
pub use game::StressLab;
pub use highlands::ROCK_FILES;
pub use metrics::{FrameMetrics, StressMetrics};
pub use observation::{StressSample, StressSnapshot, stress_observation, stress_sample};
pub use presets::{Preset, preset, presets};
pub use scene::{
    StressFile, create_workload, create_workload_with, look, set_view, stress_reflections,
    stress_shafts,
};
