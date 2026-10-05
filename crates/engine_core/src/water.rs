//! Water (decision 067): the lakes and rivers terrains' hydrology bakes
//! find (decision 066), drawn where they stand. Lakes lie still under the
//! wind's ripples, rivers run along their channels; light passing through
//! water fades with depth toward its scattered color, and its surface
//! reflects the sky and, through screen-space reflections, the scene.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::environment::{InvalidSetting, check_range};

/// Water settings: whether water draws; the color deep water scatters back
/// (linear, each 0 to 1); the meters of water through which half the light
/// passes (0.1 to 100); how strongly ripples tilt its surface (0 to 1); and
/// the meters across its largest ripples (0.1 to 100). Requires version 9.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneWater")
)]
#[serde(default, deny_unknown_fields)]
pub struct Water {
    pub enabled: bool,
    pub color: [f32; 3],
    #[cfg_attr(feature = "schema", schemars(range(min = 0.1, max = 100.0)))]
    pub clarity: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub ripples: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.1, max = 100.0)))]
    pub ripple_size: f32,
}

impl Default for Water {
    fn default() -> Self {
        Self {
            enabled: true,
            color: [0.012, 0.045, 0.05],
            clarity: 2.5,
            ripples: 0.3,
            ripple_size: 5.0,
        }
    }
}

impl Water {
    /// The first field out of range.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        for value in self.color {
            check_range("color", value, 0.0..=1.0)?;
        }
        check_range("clarity", self.clarity, 0.1..=100.0)?;
        check_range("ripples", self.ripples, 0.0..=1.0)?;
        check_range("ripple_size", self.ripple_size, 0.1..=100.0)
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
            color: [0, 1, 2]
                .map(|channel| within(self.color[channel], 0.0, 1.0, defaults.color[channel])),
            clarity: within(self.clarity, 0.1, 100.0, defaults.clarity),
            ripples: within(self.ripples, 0.0, 1.0, defaults.ripples),
            ripple_size: within(self.ripple_size, 0.1, 100.0, defaults.ripple_size),
        }
    }
}
