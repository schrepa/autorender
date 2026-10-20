//! Sky occlusion: the sky's light fades where still meshes hide the sky, so
//! caves, tunnels and buildings go dark inside away from their openings,
//! the ground under bridges and arcades darkens, and streets between tall
//! walls keep only the sky they see. No rule marks an interior: the
//! renderer looks at the sky from each visible point along a set of fixed
//! directions, through depth maps of the meshes around the camera as sun
//! shadows do, so daylight entering an opening reaches only as deep as the
//! sky can be seen from there, and the sky's reflections fade with it.
//!
//! What hides the sky: meshes of entities that stay still, while shadows
//! are enabled (it draws their casters). Nothing else does: not terrain,
//! whose own sky visibility shades it; not the objects terrain placement
//! layers place, whose covers darken the ground under them (`occupancy`);
//! not moving or skinned meshes, virtual geometry, particles or water. A
//! cave must therefore be a mesh of its own, whatever terrain it cuts into.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    /// Sky occlusion settings: the sky's light, and its reflections, fade
    /// where still meshes hide the sky, so caves, tunnels and buildings go
    /// dark inside away from their openings. Only meshes of entities that
    /// stay still hide the sky, while shadows are enabled: not terrain, not
    /// what terrain placement layers place, not moving or skinned meshes,
    /// virtual geometry, particles or water. A cave must be a mesh of its
    /// own.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneSkyOcclusion")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct SkyOcclusion {
        /// Whether the sky's light fades where still meshes hide the sky.
        pub enabled: bool = true => any,
        /// How much of the sky's light, and of its reflections, hidden sky
        /// takes away: 1 all of it, 0 none.
        pub strength: f32 = 1.0 => range(0.0, 1.0),
        /// The share of the sky's light a point keeps however little sky
        /// it sees: 0 lets the deepest cave go black; a little more stands
        /// in for light bouncing in from lit surfaces.
        pub floor: f32 = 0.0 => range(0.0, 1.0),
        /// Meters from the camera within which sharp maps decide whether
        /// meshes hide the sky, their texels a 192nd of it (0.25 m at 48 m):
        /// shorter is sharper, for small rooms and narrow openings, longer
        /// keeps more of the view sharp.
        pub near_distance: f32 = 48.0 => range(8.0, 256.0),
        /// Meters from the camera within which meshes hide the sky at all,
        /// through coarser maps beyond `near_distance` (texels a 192nd of
        /// it), fading out over a third as far again: farther, sky light is
        /// as without sky occlusion.
        pub far_distance: f32 = 400.0 => range(64.0, 4000.0),
        /// Directions of the sky looked along: 5, 9 or 13, others round down
        /// to one of them. Fewer cost less; more give smoother light near
        /// openings.
        pub directions: u32 = 13 => range(5.0, 13.0),
    }
}

impl SkyOcclusion {
    /// What a world without these settings reads as: no sky occlusion.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Whether hidden sky darkens anything: enabled with a strength.
    pub fn darkens(&self) -> bool {
        self.enabled && self.strength > 0.0
    }

    /// Directions in use: 5, 9 or 13.
    pub fn direction_count(&self) -> u32 {
        match self.directions {
            ..9 => 5,
            9..13 => 9,
            _ => 13,
        }
    }
}
