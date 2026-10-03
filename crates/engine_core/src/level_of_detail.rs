//! Level of detail settings (decision 047): meshes carry simplified levels
//! the renderer builds when they are first uploaded, and GPU-presented frames
//! draw each instance at the coarsest level whose simplification error stays
//! within `pixel_error` of the viewport height. Impostors (decision 048) are
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

/// Level of detail settings (decisions 047 and 048): whether instances draw
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
    /// Whether draw groups draw impostors where they fit (decision 048).
    pub impostors: bool,
    /// The most impostors kept baked; omitted for no budget.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_impostors: Option<u32>,
}

impl Default for LevelOfDetail {
    fn default() -> Self {
        Self {
            enabled: true,
            pixel_error: 1.0,
            impostors: true,
            max_impostors: None,
        }
    }
}

impl LevelOfDetail {
    /// Ok for a pixel error in `[0.25, 64]`.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("pixel_error", self.pixel_error, 0.25..=64.0)
    }

    /// The pixel error within `[0.25, 64]` (1 when not finite).
    pub fn sanitized(self) -> Self {
        let pixel_error = if self.pixel_error.is_finite() {
            crate::math::clamp(self.pixel_error, 0.25, 64.0)
        } else {
            1.0
        };
        Self {
            pixel_error,
            ..self
        }
    }
}
