//! Occupancy: a top-down map around the camera of what
//! placed objects cover, their tops and bottoms, which darkens the sky
//! light of what lies under or within them: forest floors and the lower
//! crowns of trees, and the ground at a boulder's foot; and a coarser one
//! reaching much farther, where forests darken their floors
//! by how much of the ground their trees cover.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::rules::{ExtraRules, settings};

settings! {
    /// Occupancy settings. Requires version 9.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneOccupancy")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Occupancy {
        /// Whether the map is drawn.
        pub enabled: bool = true => any,
        /// Meters around the camera within which covers darken, less over the
        /// outer two fifths.
        pub radius: f32 = 512.0 => range(64.0, 4096.0),
        /// The map's texels a side.
        pub resolution: u32 = 1024 => range(128, 4096),
        /// How much of the sky light full cover takes away.
        pub strength: f32 = 0.75 => range(0.0, 1.0),
        /// Meters the coarser far map reaches, with as many texels: none at
        /// or within `radius`.
        pub far_radius: f32 = 4096.0 => range(0.0, 65536.0),
    }
}

impl ExtraRules for Occupancy {}

impl Occupancy {
    /// What a world without these settings reads as: no darkening under cover.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
