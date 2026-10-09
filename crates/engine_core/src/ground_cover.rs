//! Ground cover: blades of grass and the like standing on a
//! terrain's surface materials, which renderers generate every frame around
//! the camera and never store. A surface material names the cover growing
//! where it lies ([`Cover`]); the [`GroundCover`] settings place, budget and
//! sway it. Past its radius cover grows on in tiers as long as blades stay
//! big enough on screen, and the ground beyond shows as its canopy.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::environment::{InvalidSetting, check_range};
use crate::rules::{ExtraRules, settings};

/// The cover growing where a surface material lies: blades
/// per square meter where the material lies alone, the shortest and
/// tallest blades and their width at the root (meters), the linear color
/// of their tips (roots take the ground's), and how far they lean in the
/// wind (0 stiff, 1 the most).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct Cover {
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 400.0)))]
    pub density: f32,
    pub height: [f32; 2],
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001, max = 0.25)))]
    pub width: f32,
    pub color: [f32; 3],
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub sway: f32,
}

impl Default for Cover {
    /// A meadow's grass.
    fn default() -> Self {
        Self {
            density: 90.0,
            height: [0.12, 0.4],
            width: 0.025,
            color: [0.13, 0.2, 0.05],
            sway: 0.6,
        }
    }
}

impl Cover {
    /// The first field out of range: density 0 to 400 blades per square
    /// meter, heights 0.01 to 4 meters with the shortest first, width 0.001
    /// to 0.25 meters, color 0 to 1 and sway 0 to 1.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("density", self.density, 0.0..=400.0)?;
        for height in self.height {
            check_range("height", height, 0.01..=4.0)?;
        }
        if self.height[0] > self.height[1] {
            return Err(InvalidSetting::new(
                "height",
                "must list the shortest first",
            ));
        }
        check_range("width", self.width, 0.001..=0.25)?;
        for channel in self.color {
            check_range("color", channel, 0.0..=1.0)?;
        }
        check_range("sway", self.sway, 0.0..=1.0)
    }
}

settings! {
    /// Ground cover settings. Requires version 9.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneGroundCover")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct GroundCover {
        /// Whether terrains grow their cover.
        pub enabled: bool = true => any,
        /// Meters from the camera within which cover grows densest. Past it the
        /// ground shows as its cover's canopy, and sparser tiers of wider blades
        /// reach on.
        pub radius: f32 = 40.0 => range(2.0, 200.0),
        /// How tall on screen (pixels) the average blade stands where tiers
        /// end: each tier reaches twice as far as the last, with blades sparser
        /// and wider, until its blades would stand shorter than this. 0 grows
        /// cover within the radius alone.
        pub pixels: f32 = 2.0 => range(0.0, 16.0),
        /// Scales every cover's density. Blades sway in the world's `Wind`.
        pub density: f32 = 1.0 => range(0.0, 4.0),
        /// An optional budget of blades a frame; beyond it the farthest drop.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_blades: Option<u32> = None => budget,
    }
}

impl ExtraRules for GroundCover {}

impl GroundCover {
    /// What a world without these settings reads as: no ground cover.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
