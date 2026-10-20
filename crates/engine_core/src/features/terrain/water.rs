//! Water: the lakes and rivers terrains' hydrology bakes
//! find, drawn where they stand. Lakes lie still under the
//! wind's ripples, rivers run along their channels; light passing through
//! water fades over the distance it travels to the ground behind it toward
//! its scattered color, foam gathers where it runs shallow along shores and
//! where rivers run fast, and its surface reflects the sky
//! and, through screen-space reflections, the scene.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    /// Water settings.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneWater")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Water {
        /// Whether water draws.
        pub enabled: bool = true => any,
        /// The color deep water scatters back, linear.
        pub color: [f32; 3] = [0.012, 0.045, 0.05] => range(0.0, 1.0),
        /// The meters of water through which half the light passes.
        pub clarity: f32 = 2.5 => range(0.1, 100.0),
        /// How strongly ripples tilt the surface.
        pub ripples: f32 = 0.3 => range(0.0, 1.0),
        /// The meters across the largest ripples.
        pub ripple_size: f32 = 5.0 => range(0.1, 100.0),
        /// How much foam shores and rapids gather; 0 none.
        pub foam: f32 = 0.6 => range(0.0, 1.0),
        /// The meters of water shallower than which shores foam.
        pub foam_width: f32 = 0.6 => range(0.05, 10.0),
        /// The meters across the square around the camera over which
        /// rivers run as the shallow-water equations say, from the water
        /// the bake stands there; 0 none, the bake's water standing
        /// everywhere.
        pub simulated: f32 = 0.0 => range(0.0, 1024.0),
        /// The meters across the simulation's cells.
        pub simulation_cell: f32 = 1.0 => range(0.25, 8.0),
    }
}

impl Water {
    /// What a world without these settings reads as: no water.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
