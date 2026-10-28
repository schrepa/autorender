//! Water: the lakes and rivers terrains' hydrology bakes find, drawn where
//! they stand. Rivers run along their channels as their water settles there
//! by the shallow-water equations, and lakes lie still; around the detail
//! eye, out to where its cells' unsteady features stay a few pixels across
//! ([`crate::LIVE_PIXELS`]) and as far as a budget of its cells' steps holds
//! ([`Water::live_budget`]), the same water runs live from that settled
//! state by the same equations, in cells finer than the settled water's, a
//! few across its narrowest channels, stirred by its own turbulence's
//! bursts, and hands back to it at its rim. How water moves and foams
//! follows from its state alone, what its tiles or the live solve hold at
//! each place (its depth, flow, the foam it carries, how turbulent it runs
//! and how far across its water body is), the scene's wind
//! ([`crate::Wind`]) dragging the water that runs live and the foam riding
//! it. It shows that state and nothing
//! else: its surface is the state's own, roughened where its turbulence and
//! eddies stir it, a calm mirror where nothing does, and its foam and small
//! waves particles its flow carries. Each body's look is its color, its
//! clarity and the sediment it carries with that sediment's color: light
//! passing through
//! water fades over the distance it travels to the ground behind it toward
//! its scattered color, and its surface reflects the sky and, through
//! screen-space reflections, the scene.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    /// Water settings: how terrains' water looks (its color, clarity and
    /// sediment), the cells it settles and runs live in, and how it draws;
    /// its movement and foam come from its state, the scene's wind dragging
    /// the water that runs live.
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
        /// The meters across the cells water settles in: the cook makes the
        /// settled water's records at this size; the water running live
        /// around the detail eye runs in these cells split along each side
        /// into as few as put four across the narrowest channel the
        /// terrain's bake cuts, half a meter for meter cells over a meter's
        /// samples.
        pub simulation_cell: f32 = 1.0 => range(0.25, 8.0),
        /// The most cell steps the water running live around the detail eye
        /// takes in a frame: its square holds no more cells than this over
        /// the steps of the longest frame it steps through (a thirtieth of a
        /// second, eight steps of half-meter cells), reaching nearer the eye
        /// than its cells' pixels ask where it would hold more (the cells it
        /// cuts are counted). The default holds 512 half-meter cells along a
        /// side, 256 meters.
        pub live_budget: u32 = 2_097_152 => range(65_536, 67_108_864),
        /// Points along a side of the square around the detail eye over
        /// which the foam's rafts are born and ride, half a settled cell
        /// apart: past it no foam shows. Each point costs 36 bytes on the
        /// GPU and five invocations a frame. The default reaches 256
        /// meters from the eye over meter cells, 512 across; the square
        /// holds no fewer than the waves' window around the eye needs, and
        /// the device's storage bindings may hold fewer (the points they
        /// cut are counted).
        pub foam_points: u32 = 1024 => range(64, 8192),
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
