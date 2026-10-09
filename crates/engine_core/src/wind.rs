//! The world's wind: one velocity and gust strength that
//! everything swaying in it reads, ground cover's blades and placed
//! objects alike. Renderers sway them by it at the frame's presentation
//! time, and temporal frames by the previous frame's for their motion.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::environment::{InvalidSetting, check_range};

/// The wind: its horizontal velocity (x and z, meters per second, each -40
/// to 40) and how strongly gusts traveling with it vary its push, 0 to 1.
/// Requires version 9.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneWind")
)]
#[serde(default, deny_unknown_fields)]
pub struct Wind {
    pub velocity: [f32; 2],
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub gusts: f32,
}

impl Default for Wind {
    fn default() -> Self {
        Self {
            velocity: [2.5, 1.0],
            gusts: 0.45,
        }
    }
}

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

    /// The first field out of range.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        for speed in self.velocity {
            check_range("velocity", speed, -40.0..=40.0)?;
        }
        check_range("gusts", self.gusts, 0.0..=1.0)
    }

    /// Every value in range.
    pub fn sanitized(self) -> Self {
        let finite = |value: f32| if value.is_finite() { value } else { 0.0 };
        Self {
            velocity: self
                .velocity
                .map(|speed| crate::math::clamp(finite(speed), -40.0, 40.0)),
            gusts: crate::math::clamp(finite(self.gusts), 0.0, 1.0),
        }
    }
}
