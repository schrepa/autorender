//! How foliage looks: the leaves and needles of masked materials (and
//! the impostor cards baked from them) and ground cover's blades.
//! Transmission, variation and the ground blend start on; hashed cutoffs
//! and crown depth start off.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    /// Foliage looks.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneFoliage")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Foliage {
        /// Sunlight leaves, needles and blades pass toward the eye, seen
        /// against the sun through them, tinted by their color and shaded
        /// as the sun is; 0 turns it off.
        pub transmission: f32 = 1.0 => range(0.0, 4.0),
        /// How far each plant's leaves stray from its material's color, by
        /// a hash of where it stands, along a gradient from drier to deeper
        /// green; bark keeps its color. 0 turns it off.
        pub variation: f32 = 1.0 => range(0.0, 1.0),
        /// Whether masked leaves dissolve over their texture's coverage on
        /// temporally antialiased frames, a threshold turning across pixels
        /// and frames instead of the material's cutoff.
        pub hashed_cutoff: bool = false => any,
        /// Meters up from the ground over which plants' color and light
        /// blend into the ground's at their base; 0 turns it off.
        pub ground_blend: f32 = 0.4 => range(0.0, 4.0),
        /// Meters into a crown over which the sun's light halves: masked
        /// leaves and needles, and the cards baked from them, take the sun
        /// through their crown from its sun-facing side, down along the
        /// sun and across from where the crown meets sunlit air, instead
        /// of the first caster toward the sun shadowing them whole; 0
        /// turns it off.
        pub crown_depth: f32 = 0.0 => range(0.0, 8.0),
    }
}

impl Foliage {
    /// What a world without these settings reads as: every look off.
    pub fn off() -> Self {
        Self {
            transmission: 0.0,
            variation: 0.0,
            hashed_cutoff: false,
            ground_blend: 0.0,
            crown_depth: 0.0,
        }
    }

    /// Whether any look is on.
    pub fn any(&self) -> bool {
        self.transmission > 0.0
            || self.variation > 0.0
            || self.hashed_cutoff
            || self.ground_blend > 0.0
            || self.crown_depth > 0.0
    }
}
