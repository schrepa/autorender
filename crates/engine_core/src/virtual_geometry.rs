//! Virtual geometry settings (decision 044): meshes built from cluster
//! hierarchies ([`crate::Mesh::from_virtual`]) draw the clusters each view
//! needs, selected on the GPU, instead of their fallback cut.

use bevy_ecs::prelude::Resource;

/// Virtual geometry settings, inserted by default. Without this resource,
/// or when disabled, meshes with virtual geometry draw their fallback cut.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct VirtualGeometry {
    pub enabled: bool,
    /// The largest simplification error a drawn cluster may show, in pixels
    /// of the viewport height: smaller draws more triangles.
    pub pixel_error: f32,
    /// An optional budget of pages (128 KiB each) kept resident on the GPU;
    /// beyond it the least recently used make room, and pinned pages always
    /// stay. Without one, only the device's limits bound them.
    pub max_resident_pages: Option<u32>,
}

impl Default for VirtualGeometry {
    fn default() -> Self {
        Self {
            enabled: true,
            pixel_error: 1.5,
            max_resident_pages: None,
        }
    }
}

impl VirtualGeometry {
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
        }
    }
}
