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

use crate::environment::{InvalidSetting, check_range};

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
    pub enabled: bool,
    /// The largest simplification error a drawn level may show, in pixels of
    /// the viewport height, from 0.25 to 64: smaller draws finer levels
    /// farther away.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.25, max = 64.0)))]
    pub pixel_error: f32,
    /// Whether draw groups draw impostors where they fit.
    pub impostors: bool,
    /// The most impostors kept baked; omitted for no budget.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_impostors: Option<u32>,
    /// Seconds placed objects take to fade between how the camera shows
    /// them, from 0 (at once) to 4.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 4.0)))]
    pub fade_seconds: f32,
}

impl Default for LevelOfDetail {
    fn default() -> Self {
        Self {
            enabled: true,
            pixel_error: 1.0,
            impostors: true,
            max_impostors: None,
            fade_seconds: 0.5,
        }
    }
}

impl LevelOfDetail {
    /// Ok for a pixel error in `[0.25, 64]` and a fade in `[0, 4]` seconds.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("pixel_error", self.pixel_error, 0.25..=64.0)?;
        check_range("fade_seconds", self.fade_seconds, 0.0..=4.0)
    }

    /// The pixel error within `[0.25, 64]` (1 when not finite) and the fade
    /// within `[0, 4]` seconds (0.5 when not finite).
    pub fn sanitized(self) -> Self {
        let pixel_error = if self.pixel_error.is_finite() {
            crate::math::clamp(self.pixel_error, 0.25, 64.0)
        } else {
            1.0
        };
        let fade_seconds = if self.fade_seconds.is_finite() {
            crate::math::clamp(self.fade_seconds, 0.0, 4.0)
        } else {
            0.5
        };
        Self {
            pixel_error,
            fade_seconds,
            ..self
        }
    }
}
