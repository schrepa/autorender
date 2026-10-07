//! Contact shadows (decision 062): short rays toward the sun through the
//! depth buffer find what shadow maps are too coarse to hold, such as grass
//! blades on the ground, a pebble's or a crate's foot and the gaps between
//! stones, and darken the sun's share of their light.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::environment::{InvalidSetting, check_range};

/// Contact shadow settings: whether they trace, how far each ray reaches
/// toward the sun in meters (0.05 to 4), how much of the sun's light a hit
/// takes away (0 to 1; 0 turns them off), and the meters from the camera
/// within which surfaces trace, their shadows fading out over its last
/// third (decision 098; 1 to 1000). Requires version 9.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneContactShadows")
)]
#[serde(default, deny_unknown_fields)]
pub struct ContactShadows {
    pub enabled: bool,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.05, max = 4.0)))]
    pub length: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub strength: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 1.0, max = 1000.0)))]
    pub distance: f32,
}

impl Default for ContactShadows {
    fn default() -> Self {
        Self {
            enabled: true,
            length: 0.6,
            strength: 0.85,
            distance: 80.0,
        }
    }
}

impl ContactShadows {
    /// The first field out of range.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("length", self.length, 0.05..=4.0)?;
        check_range("strength", self.strength, 0.0..=1.0)?;
        check_range("distance", self.distance, 1.0..=1000.0)
    }

    /// Every value in range: out-of-range values clamp, unusable ones
    /// fall back to the defaults.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        let within = |value: f32, low: f32, high: f32, fallback: f32| {
            if value.is_finite() {
                crate::math::clamp(value, low, high)
            } else {
                fallback
            }
        };
        Self {
            enabled: self.enabled,
            length: within(self.length, 0.05, 4.0, defaults.length),
            strength: within(self.strength, 0.0, 1.0, 0.0),
            distance: within(self.distance, 1.0, 1000.0, defaults.distance),
        }
    }

    /// Whether rays trace: enabled with a strength.
    pub fn traces(&self) -> bool {
        self.enabled && self.strength > 0.0
    }
}
