//! Ground cover: blades of grass and the like standing on a
//! terrain's surface materials, which renderers generate every frame around
//! the camera and never store. A surface material names the cover growing
//! where it lies ([`Cover`]); the [`GroundCover`] settings place, budget and
//! sway it. As blades thin with distance each one left stands as a tuft of
//! the leaves of those it replaces, and blades and tufts grow taller or
//! shorter, lean, and take their hue and dryness by the clump they stand in.
//! Past its radius cover grows on in tiers as long as tufts stay wide enough
//! on screen, retiring in a fixed order as they narrow, and the ground shows
//! the leaves they leave as its canopy, their average, clump by clump.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::environment::InvalidSetting;
use engine_settings::{ExtraRules, settings};

settings! {
    extra_rules;
    /// The cover growing where a surface material lies.
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct Cover {
        /// Blades per square meter where the material lies alone, on ground
        /// halfway between dry and lush: lusher ground grows more, drier
        /// ground fewer.
        pub density: f32 = 90.0 => range(0.0, 400.0),
        /// The shortest and tallest blades, meters, shortest first, on
        /// ground halfway between dry and lush: lusher ground grows them
        /// taller, drier ground shorter.
        pub height: [f32; 2] = [0.12, 0.4] => range(0.01, 4.0),
        /// Blade width at the root, meters. As blades thin with distance each
        /// one left stands as a tuft as wide at its roots in all as the
        /// blades it replaces, in leaves as wide as their roots, or up to a
        /// tenth as wide as they are tall, as grass leaves are.
        pub width: f32 = 0.025 => range(0.001, 0.25),
        /// The linear color of the blades' tips; roots take the ground's.
        /// Lusher ground darkens and greens them, drier ground pales them.
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
        /// Meters from the camera within which cover grows densest, its blades
        /// thinning past a fifth of it, each one left a tuft holding the
        /// leaves of those it stands for. Past the radius sparser tiers of
        /// tufts as wide as there reach on, and the ground shows the leaves
        /// they leave as its cover's canopy.
        pub radius: f32 = 40.0 => range(2.0, 200.0),
        /// How wide on screen (pixels) tufts stand as they retire: each
        /// retires, in a fixed order, once it spans fewer than this times a
        /// factor of its own from a half to two, and tiers, each reaching
        /// twice as far as the last, end where the last has. 0 grows cover
        /// within the radius alone.
        pub pixels: f32 = 1.0 => range(0.0, 16.0),
        /// Scales every cover's density. Blades sway in the world's `Wind`.
        pub density: f32 = 1.0 => range(0.0, 4.0),
        /// An optional budget of blades and tufts a frame; beyond it the
        /// farthest drop.
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
