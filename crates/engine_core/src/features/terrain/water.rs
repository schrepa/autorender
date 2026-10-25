//! Water: the lakes and rivers terrains' hydrology bakes find, drawn where
//! they stand. Rivers run along their channels as their water settles there
//! by the shallow-water equations, near the camera and far from it, and
//! lakes lie still. How water moves and foams follows from its state alone,
//! what its tiles or the live simulation hold at each place (its depth,
//! flow, the foam it carries, how turbulent it runs and how far across its
//! water body is), under the scene's wind ([`crate::Wind`]); each body's
//! look is its color, its clarity and the sediment it carries with that
//! sediment's color: light passing through water fades over the distance it
//! travels to the ground behind it toward its scattered color, and its
//! surface reflects the sky and, through screen-space reflections, the
//! scene.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    /// Water settings: how terrains' water looks (its color, clarity and
    /// sediment) and how it draws and runs live; its movement and foam come
    /// from its state, under the scene's wind.
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
        /// How strongly the fine sediment water carries scatters light, per
        /// meter: light passing through fades by it as well as by what the
        /// water's clarity takes, and the water's body scatters back
        /// `sediment_color` by the sediment's share of the two; 0 none.
        pub sediment: f32 = 0.15 => range(0.0, 50.0),
        /// The color the water's sediment scatters back, linear.
        pub sediment_color: [f32; 3] = [0.05, 0.065, 0.03] => range(0.0, 1.0),
        /// The meters across the square around the camera over which
        /// rivers run live as the shallow-water equations say, in cells
        /// `live_cell` across, starting from their water as it settled
        /// there and taking the settled water's past its edges; 0 none, the
        /// settled water standing everywhere.
        pub simulated: f32 = 0.0 => range(0.0, 1024.0),
        /// The meters across the cells water settles in: the cook makes the
        /// settled water's records at this size.
        pub simulation_cell: f32 = 1.0 => range(0.25, 8.0),
        /// The meters across the live window's cells: the square
        /// `simulated` asks for runs in cells this size, each starting from
        /// the settled water's cells around it.
        pub live_cell: f32 = 0.5 => range(0.25, 8.0),
        /// Pixels the water's drawn triangles may stray from its surface,
        /// seen from the detail eye: each terrain's water subdivides a tree
        /// of its own, splitting its leaves straying more and merging those
        /// whose halves would not, as the ground's does. Smaller draws more
        /// triangles. The water draws as many pixels over its surface as
        /// this and the ground's own error together, so the ground drawn
        /// under it never hides it.
        pub subdivision_error: f32 = 1.0 => range(0.25, 64.0),
        /// The most leaves each terrain's water's subdivision draws, 28
        /// bytes each on the GPU: past them leaves split no more, counted,
        /// and the water there stays coarser.
        pub subdivision_leaves: u32 = 65_536 => range(1024, 4_000_000),
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
