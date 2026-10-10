//! Deterministic stress workloads built exclusively through the public engine SDK.
mod authored;
#[cfg(target_arch = "wasm32")]
mod browser;
mod catalog;
mod caves;
mod city;
mod config;
mod dense;
mod game;
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
pub use game::StressLab;
pub use highlands::{FILES as HIGHLANDS_FILES, ROCK_FILES};
pub use metrics::{FrameMetrics, StressMetrics};
pub use observation::{StressSample, StressSnapshot, stress_observation, stress_sample};
pub use scene::{
    StressFile, create_workload, create_workload_with, look, set_view, stress_reflections,
    stress_shafts,
};
