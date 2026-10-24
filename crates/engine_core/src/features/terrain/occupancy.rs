//! Occupancy: top-down maps around the detail eye of what
//! placed objects cover, their tops and bottoms, which darken the sky
//! light of what lies under or within them: forest floors and the lower
//! crowns of trees, and the ground at a boulder's foot. The maps nest, each
//! reaching twice as far as the one inside it with texels twice as wide, so
//! forests far away darken their floors by how much of the ground their
//! trees cover, as near ones do.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

/// Most occupancy maps ([`Occupancy::levels`]).
pub const MAX_OCCUPANCY_LEVELS: u32 = 8;

settings! {
    /// Occupancy settings.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneOccupancy")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Occupancy {
        /// Whether the maps are drawn.
        pub enabled: bool = true => any,
        /// Meters around the detail eye the finest map holds: each coarser
        /// one reaches twice as far.
        pub radius: f32 = 512.0 => range(64.0, 4096.0),
        /// Each map's texels a side, a multiple of 32: the cells its window
        /// follows the eye by, sixteen a side, span whole texels, and each
        /// cell of a finer map whole texels of the next.
        pub resolution: u32 = 1024 => multiple(32, 128, 4096),
        /// Maps around the detail eye, the finest first, each read where a
        /// pixel's size picks it: covers darken out to `radius` times two to
        /// one less than their count, fading out over the outer half of the
        /// coarsest map's reach.
        pub levels: u32 = 4 => range(1, MAX_OCCUPANCY_LEVELS),
        /// How much of the sky light full cover takes away.
        pub strength: f32 = 0.75 => range(0.0, 1.0),
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
}
