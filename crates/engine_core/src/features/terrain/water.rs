//! Water: the lakes and rivers terrains' hydrology bakes
//! find, drawn where they stand. Lakes lie still under the
//! wind's ripples, rivers run along their channels as their water settles
//! there by the shallow-water equations, near the camera and far from it;
//! light passing through water fades over the distance it travels to the
//! ground behind it toward its scattered color, foam gathers where it runs
//! shallow along shores and where the settled water churns it up, and its
//! surface reflects the sky and, through screen-space reflections, the
//! scene.

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
        /// rivers run live as the shallow-water equations say, starting
        /// from their water as it settled there and taking the settled
        /// water's past its edges; 0 none, the settled water standing
        /// everywhere.
        pub simulated: f32 = 0.0 => range(0.0, 1024.0),
        /// The meters across the cells water settles and is simulated in:
        /// the cook makes the settled water's records at this size.
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
