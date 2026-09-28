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
}

impl Default for VirtualGeometry {
    fn default() -> Self {
        Self {
            enabled: true,
            pixel_error: 1.5,
        }
    }
}

impl VirtualGeometry {
    /// The pixel error within `[0.25, 64]` (1.5 when not finite).
    pub fn sanitized(self) -> Self {
        let pixel_error = if self.pixel_error.is_finite() {
            crate::math::clamp(self.pixel_error, 0.25, 64.0)
        } else {
            1.5
        };
        Self {
            enabled: self.enabled,
            pixel_error,
        }
    }
}
