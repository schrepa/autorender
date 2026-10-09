//! Level of detail settings: meshes carry simplified levels
//! the renderer builds when they are first uploaded, and GPU-presented frames
//! draw each instance at the coarsest level whose simplification error stays
//! within `pixel_error` of the viewport height. Impostors are
//! the coarsest level: cards showing a draw group's mesh baked from many
//! directions, for instances too small to show more.
//!
//! The [`LevelOfDetail`] settings are inserted by default; without them, or
//! when disabled, every instance draws its mesh itself. Opaque draw groups
//! whose meshes have enough triangles draw impostors where those stay within
//! `pixel_error`; without `max_impostors`, only the GPU's texture array
//! layers bound how many stay baked, and draw groups beyond a budget draw
//! their meshes.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::{ExtraRules, settings};

settings! {
    /// Level of detail settings: whether instances draw
    /// simplified levels of their meshes, and impostors, where the
    /// simplification stays within a pixel error. Requires version 8.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneLevelOfDetail")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct LevelOfDetail {
        /// Whether instances draw simplified levels.
        pub enabled: bool = true => any,
        /// The largest simplification error a drawn level may show, in pixels
        /// of the viewport height: smaller draws finer levels farther away.
        pub pixel_error: f32 = 1.0 => range(0.25, 64.0),
        /// Whether draw groups draw impostors where they fit.
        pub impostors: bool = true => any,
        /// The most impostors kept baked; omitted for no budget.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_impostors: Option<u32> = None => budget,
        /// Seconds placed objects take to fade between how the camera shows
        /// them, from 0 (at once).
        pub fade_seconds: f32 = 0.5 => range(0.0, 4.0),
    }
}

impl ExtraRules for LevelOfDetail {}

impl LevelOfDetail {
    /// What a world without these settings reads as: every instance at full detail.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
