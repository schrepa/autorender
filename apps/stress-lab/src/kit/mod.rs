//! The content kit: what the Stress Lab's worlds build with and share, so
//! no world reaches into another's code, and what they read from and hand
//! the workload. A world module imports only the kit, the engine and itself
//! (lint's content locality stage).

pub(crate) mod field;
pub(crate) mod model_builder;
pub(crate) mod motion;
pub(crate) mod plan;
pub(crate) mod random;
pub(crate) mod shape_builder;

// What a world reads, implements and hands the workload.
pub(crate) use crate::StressConfig;
pub(crate) use crate::scene::{StressFile, Workload};
pub(crate) use crate::world::StressWorld;
