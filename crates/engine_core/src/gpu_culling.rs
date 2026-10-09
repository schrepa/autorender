//! Culling of GPU-presented frames beyond the frustum: occlusion culling,
//! which leaves what the last frame's depths hide undrawn, and sub-pixel
//! culling, which leaves what covers no pixel center undrawn. Both change no
//! output; they run where culling runs without an instance budget.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::environment::InvalidSetting;

/// Occlusion culling settings: whether GPU-presented frames draw what the
/// last frame showed, test every other frustum-visible instance against a
/// depth pyramid of those depths, then draw only what it does not hide.
/// Output is unchanged; hidden instances cost no drawing. On by default.
/// Requires version 9.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneOcclusionCulling")
)]
#[serde(default, deny_unknown_fields)]
pub struct OcclusionCulling {
    pub enabled: bool,
}

impl Default for OcclusionCulling {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl OcclusionCulling {
    /// Every value is valid.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        Ok(())
    }

    pub fn sanitized(self) -> Self {
        self
    }
}

/// Sub-pixel culling settings: whether GPU-presented frames leave visible
/// instances whose projected bounds cover no pixel center undrawn. They
/// could produce no fragments, so output is unchanged. On by default.
/// Requires version 9.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneSubpixelCulling")
)]
#[serde(default, deny_unknown_fields)]
pub struct SubpixelCulling {
    pub enabled: bool,
}

impl Default for SubpixelCulling {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl SubpixelCulling {
    /// Every value is valid.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        Ok(())
    }

    pub fn sanitized(self) -> Self {
        self
    }
}
