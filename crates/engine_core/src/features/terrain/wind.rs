//! The world's wind: one velocity and gust strength that
//! everything swaying in it reads, ground cover's blades and placed
//! objects alike. Renderers sway them by it at the frame's presentation
//! time, and temporal frames by the previous frame's for their motion;
//! bodies feel the same gusts ([`Wind::at`]) at the fixed tick's.

use bevy_ecs::prelude::Resource;
use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::math::sin;
use engine_settings::settings;

/// Gusts, as two waves running with the wind that `wind.wgsl` sways
/// placed objects by: each wave's share of the gust strength, its rate
/// (radians a second) and how quickly it varies along the wind (radians a
/// meter); the quicker wave also varies across x.
pub const GUST_SLOW_SHARE: f32 = 0.6;
pub const GUST_SLOW_RATE: f32 = 0.9;
pub const GUST_SLOW_TRAVEL: f32 = 0.05;
pub const GUST_QUICK_SHARE: f32 = 0.4;
pub const GUST_QUICK_RATE: f32 = 2.3;
pub const GUST_QUICK_TRAVEL: f32 = 0.17;
pub const GUST_QUICK_ACROSS: f32 = 0.31;

settings! {
    /// The wind.
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

    /// The wind's horizontal velocity (x and z) at world `(x, z)` at
    /// `seconds`, gusts and all: what sways the placed objects there.
    pub fn at(&self, x: f32, z: f32, seconds: f32) -> Vec2 {
        let velocity = Vec2::from_array(self.velocity);
        let speed = velocity.length();
        if speed < 1.0e-4 {
            return Vec2::ZERO;
        }
        let toward = velocity / speed;
        let travel = x * toward.x + z * toward.y;
        let gust = 1.0
            + self.gusts
                * (GUST_SLOW_SHARE * sin(seconds * GUST_SLOW_RATE - travel * GUST_SLOW_TRAVEL)
                    + GUST_QUICK_SHARE
                        * sin(seconds * GUST_QUICK_RATE - travel * GUST_QUICK_TRAVEL
                            + x * GUST_QUICK_ACROSS));
        velocity * gust
    }
}
