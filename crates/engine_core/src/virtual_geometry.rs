//! Virtual geometry settings (decision 044): meshes built from cluster
//! hierarchies ([`crate::Mesh::from_virtual`]) draw the clusters each view
//! needs, selected on the GPU, instead of their fallback cut.
//!
//! The [`VirtualGeometry`] settings are inserted by default; without them,
//! or when disabled, meshes with virtual geometry draw their fallback cut.
//! Resident pages live on the GPU. The camera's clusters shade as they draw;
//! with the visibility buffer (opt-in), where the device binds enough storage
//! buffers, they shade once per pixel after all of them are drawn.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::environment::{InvalidSetting, check_budget, check_range};

/// Virtual geometry settings (decision 044): whether meshes built as cluster
/// hierarchies draw the clusters each view needs, and how coarse those may
/// be. Requires version 8.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneVirtualGeometry")
)]
#[serde(default, deny_unknown_fields)]
pub struct VirtualGeometry {
    pub enabled: bool,
    /// The largest simplification error a drawn cluster may show, in pixels
    /// of the viewport height, from 0.25 to 64: smaller draws more
    /// triangles.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.25, max = 64.0)))]
    pub pixel_error: f32,
    /// An optional budget of streamed pages (128 KiB each) kept resident;
    /// beyond it the least recently used make room, and pinned pages always
    /// stay. Omitted, only the device's limits bound them.
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_resident_pages: Option<u32>,
    /// Whether the camera's clusters shade through the visibility buffer
    /// (decision 045), each pixel once; false shades them as they draw,
    /// which was faster on the integrated GPU measured. Omitted, false.
    #[serde(skip_serializing_if = "is_false")]
    pub visibility_buffer: bool,
}

/// Scenes omit the visibility buffer while it is off.
fn is_false(value: &bool) -> bool {
    !*value
}

impl Default for VirtualGeometry {
    fn default() -> Self {
        Self {
            enabled: true,
            pixel_error: 1.5,
            max_resident_pages: None,
            visibility_buffer: false,
        }
    }
}

impl VirtualGeometry {
    /// Ok for a pixel error in `[0.25, 64]` and a budget of at least one
    /// page.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("pixel_error", self.pixel_error, 0.25..=64.0)?;
        check_budget("max_resident_pages", self.max_resident_pages)
    }

    /// The pixel error within `[0.25, 64]` (1.5 when not finite), and a
    /// budget of at least one page.
    pub fn sanitized(self) -> Self {
        let pixel_error = if self.pixel_error.is_finite() {
            crate::math::clamp(self.pixel_error, 0.25, 64.0)
        } else {
            1.5
        };
        Self {
            enabled: self.enabled,
            pixel_error,
            max_resident_pages: self.max_resident_pages.map(|pages| pages.max(1)),
            visibility_buffer: self.visibility_buffer,
        }
    }
}
