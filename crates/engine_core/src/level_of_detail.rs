//! Level of detail settings (decision 047): meshes carry simplified levels
//! the renderer builds when they are first uploaded, and GPU-presented frames
//! draw each instance at the coarsest level whose simplification error stays
//! within `pixel_error` of the viewport height.

use bevy_ecs::prelude::Resource;

/// Level of detail settings, inserted by default. Without this resource, or
/// when disabled, every instance draws its mesh itself.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct LevelOfDetail {
    pub enabled: bool,
    /// The largest simplification error a drawn level may show, in pixels
    /// of the viewport height: smaller draws finer levels farther away.
    pub pixel_error: f32,
}

impl Default for LevelOfDetail {
    fn default() -> Self {
        Self {
            enabled: true,
            pixel_error: 1.0,
        }
    }
}

impl LevelOfDetail {
    /// The pixel error within `[0.25, 64]` (1 when not finite).
    pub fn sanitized(self) -> Self {
        let pixel_error = if self.pixel_error.is_finite() {
            crate::math::clamp(self.pixel_error, 0.25, 64.0)
        } else {
            1.0
        };
        Self {
            enabled: self.enabled,
            pixel_error,
        }
    }
}
