//! Occupancy: a top-down map around the camera of what
//! placed objects cover, their tops and bottoms, which darkens the sky
//! light of what lies under or within them: forest floors and the lower
//! crowns of trees, and the ground at a boulder's foot; and a coarser one
//! reaching much farther, where forests darken their floors
//! by how much of the ground their trees cover.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::environment::{InvalidSetting, check_range};

/// Occupancy settings: whether the map is drawn, the meters around the
/// camera within which covers darken, less over the outer two fifths
/// (64 to 4096), the map's texels a side (128 to 4096), how
/// much of the sky light full cover takes away (0 to 1), and the meters
/// the coarser far map reaches (0, none, or the radius to
/// 65536), its texels as many. Requires version 9.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneOccupancy")
)]
#[serde(default, deny_unknown_fields)]
pub struct Occupancy {
    pub enabled: bool,
    #[cfg_attr(feature = "schema", schemars(range(min = 64.0, max = 4096.0)))]
    pub radius: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 128, max = 4096)))]
    pub resolution: u32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub strength: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 65536.0)))]
    pub far_radius: f32,
}

impl Default for Occupancy {
    fn default() -> Self {
        Self {
            enabled: true,
            radius: 512.0,
            resolution: 1024,
            strength: 0.75,
            far_radius: 4096.0,
        }
    }
}

impl Occupancy {
    /// What a world without these settings reads as: no darkening under cover.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// The first field out of range.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("radius", self.radius, 64.0..=4096.0)?;
        if !(128..=4096).contains(&self.resolution) {
            return Err(InvalidSetting::new("resolution", "must be in [128, 4096]"));
        }
        check_range("strength", self.strength, 0.0..=1.0)?;
        check_range("far_radius", self.far_radius, 0.0..=65536.0)
    }

    /// Every value in range.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        let finite = |value: f32, fallback: f32| if value.is_finite() { value } else { fallback };
        Self {
            enabled: self.enabled,
            radius: crate::math::clamp(finite(self.radius, defaults.radius), 64.0, 4096.0),
            resolution: self.resolution.clamp(128, 4096),
            strength: crate::math::clamp(finite(self.strength, defaults.strength), 0.0, 1.0),
            far_radius: crate::math::clamp(
                finite(self.far_radius, defaults.far_radius),
                0.0,
                65536.0,
            ),
        }
    }
}
