//! Level of detail settings (decision 047): meshes carry simplified levels
//! the renderer builds when they are first uploaded, and GPU-presented frames
//! draw each instance at the coarsest level whose simplification error stays
//! within `pixel_error` of the viewport height. Impostors (decision 048) are
//! the coarsest level: cards showing a draw group's mesh baked from many
//! directions, for instances too small to show more.

use bevy_ecs::prelude::Resource;

/// Level of detail settings, inserted by default. Without this resource, or
/// when disabled, every instance draws its mesh itself.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct LevelOfDetail {
    pub enabled: bool,
    /// The largest simplification error a drawn level may show, in pixels
    /// of the viewport height: smaller draws finer levels farther away.
    pub pixel_error: f32,
    /// Whether opaque draw groups whose meshes have enough triangles draw
    /// their impostors where those stay within `pixel_error` (decision 048).
    pub impostors: bool,
    /// The most impostors kept baked, `None` for no budget beyond the GPU's
    /// texture array layers; draw groups beyond it draw their meshes.
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
