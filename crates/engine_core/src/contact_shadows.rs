//! Contact shadows: short rays toward the sun through the
//! depth buffer find what shadow maps are too coarse to hold, such as grass
//! blades on the ground, a pebble's or a crate's foot and the gaps between
//! stones, and darken the sun's share of their light.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    /// Contact shadow settings. Requires version 9.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneContactShadows")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct ContactShadows {
        /// Whether contact shadows trace.
        pub enabled: bool = true => any,
        /// How far each ray reaches toward the sun, in meters.
        pub length: f32 = 0.6 => range(0.05, 4.0),
        /// How much of the sun's light a hit takes away; 0 turns them off.
        pub strength: f32 = 0.85 => range(0.0, 1.0),
        /// Meters from the camera within which surfaces trace, their shadows
        /// fading out over its last third.
        pub distance: f32 = 80.0 => range(1.0, 1000.0),
    }
}

impl ContactShadows {
    /// What a world without these settings reads as: no contact shadows.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Whether rays trace: enabled with a strength.
    pub fn traces(&self) -> bool {
        self.enabled && self.strength > 0.0
    }
}
