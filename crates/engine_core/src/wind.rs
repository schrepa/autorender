//! The world's wind: one velocity and gust strength that
//! everything swaying in it reads, ground cover's blades and placed
//! objects alike. Renderers sway them by it at the frame's presentation
//! time, and temporal frames by the previous frame's for their motion.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::rules::{ExtraRules, settings};

settings! {
    /// The wind. Requires version 9.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneWind")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Wind {
        /// Horizontal velocity: x and z, meters per second.
        pub velocity: [f32; 2] = [2.5, 1.0] => range(-40.0, 40.0),
        /// How strongly gusts traveling with the wind vary its push.
        pub gusts: f32 = 0.45 => range(0.0, 1.0),
    }
}

impl ExtraRules for Wind {}

impl Wind {
    /// What a world without these settings reads as: still air.
    pub fn off() -> Self {
        Self::STILL
    }

    /// Still air: what a world without the resource reads.
    pub const STILL: Self = Self {
        velocity: [0.0; 2],
        gusts: 0.0,
    };
}
