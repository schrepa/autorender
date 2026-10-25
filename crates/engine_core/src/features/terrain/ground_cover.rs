//! Ground cover: blades of grass and the like standing on a
//! terrain's surface materials, which renderers generate every frame around
//! the camera and never store. A surface material names the cover growing
//! where it lies ([`Cover`]); the [`GroundCover`] settings place, budget and
//! sway it. Blades stand by their size on screen alone: as they narrow, a
//! share of them stands in a fixed order, each widened by its inverse so
//! they cover as much ground, and their colors drawn toward their canopy's
//! so they vary as much; as they grow short on screen they retire in that
//! order, and the ground shows the leaves they leave as its canopy, what
//! they show on average.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::environment::InvalidSetting;
use engine_settings::{ExtraRules, settings};

/// The pixels a pattern's features must span to draw steadily under
/// temporal antialiasing. Its frames sample each pixel up to half a pixel
/// off its middle, which moves the phase at which a pixel samples a pattern
/// `p` pixels long by up to `1 / (2 p)` of its period from frame to frame:
/// no more than an eighth of it where `p` is 4. Finer patterns draw as
/// their average: waves' slopes as roughness through their variance, blades
/// as their canopy, foam's bubbles as their expected cover. Shaders mirror
/// it (`STEADY_PIXELS`), and blades span it at least by default
/// ([`GroundCover::pixels`]).
pub const STEADY_PIXELS: f32 = 4.0;

settings! {
    extra_rules;
    /// The cover growing where a surface material lies.
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct Cover {
        /// Blades per square meter where the material lies alone.
        pub density: f32 = 90.0 => range(0.0, 400.0),
        /// The shortest and tallest blades, meters, shortest first.
        pub height: [f32; 2] = [0.12, 0.4] => range(0.01, 4.0),
        /// Blade width at the root, meters.
        pub width: f32 = 0.025 => range(0.001, 0.25),
        /// The linear color of the blades' tips; roots take the ground's.
        pub color: [f32; 3] = [0.13, 0.2, 0.05] => range(0.0, 1.0),
        /// How far they lean in the wind: 0 stiff, 1 the most.
        pub sway: f32 = 0.6 => range(0.0, 1.0),
    }
}

impl ExtraRules for Cover {
    /// The shortest blades come first.
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.height[0] <= self.height[1] {
            Ok(())
        } else {
            Err(InvalidSetting::new(
                "height",
                "must list the shortest first",
            ))
        }
    }

    fn sanitize_extra(self) -> Self {
        let [low, high] = self.height;
        Self {
            height: [crate::math::min(low, high), crate::math::max(low, high)],
            ..self
        }
    }
}

settings! {
    /// Ground cover settings.
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
        /// The pixels on screen blades span at least while they stand,
        /// across and up: where blades as wide as their roots would span
        /// fewer across, a share of them stands in a fixed order, each
        /// widened by its inverse to span these pixels, so they cover as
        /// much ground, their colors drawn toward their canopy's by the root
        /// of the share standing; and once their height spans fewer than
        /// these times a factor of their own from a half to two, they retire
        /// in that order, the ground showing them as its cover's canopy, so
        /// cover reaches as far as its last blade stands that tall. By
        /// default the pixels temporal antialiasing draws steadily
        /// (`STEADY_PIXELS`); twice the pixels stand about a quarter as many
        /// blades for the same cover.
        pub pixels: f32 = STEADY_PIXELS => range(0.25, 16.0),
        /// Scales every cover's density. Blades sway in the world's `Wind`.
        pub density: f32 = 1.0 => range(0.0, 4.0),
        /// An optional budget of blades a frame: where the view would take
        /// in more, the blades that thin stand wider, fewer covering as much
        /// ground; any past it still drop and count.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_blades: Option<u32> = None => budget,
    }
}

impl GroundCover {
    /// What a world without these settings reads as: no ground cover.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
