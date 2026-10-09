//! Virtual geometry settings: meshes built from cluster
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

use crate::rules::{ExtraRules, settings};

settings! {
    /// Virtual geometry settings: whether meshes built as cluster
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
        /// Whether meshes with clusters draw as virtual geometry.
        pub enabled: bool = true => any,
        /// The largest simplification error a drawn cluster may show, in pixels
        /// of the viewport height: smaller draws more triangles.
        pub pixel_error: f32 = 1.5 => range(0.25, 64.0),
        /// An optional budget of streamed pages (128 KiB each) kept resident;
        /// beyond it the least recently used make room, and pinned pages always
        /// stay. Omitted, only the device's limits bound them.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_resident_pages: Option<u32> = None => budget,
        /// Whether the camera's clusters shade through the visibility buffer,
        /// each pixel once; false shades them as they draw,
        /// which was faster on the integrated GPU measured. Omitted, false.
        #[serde(skip_serializing_if = "is_false")]
        pub visibility_buffer: bool = false => any,
    }
}

impl ExtraRules for VirtualGeometry {}

/// Scenes omit the visibility buffer while it is off.
fn is_false(value: &bool) -> bool {
    !*value
}

impl VirtualGeometry {
    /// What a world without these settings reads as: fallback cuts.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
