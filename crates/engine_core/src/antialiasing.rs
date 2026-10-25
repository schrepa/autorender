//! Temporal antialiasing: ordinary GPU-presented frames rasterize with a
//! sub-pixel jitter and blend with the previous frame's output, reprojected
//! by per-pixel motion. Development steps, renders, previews and captures
//! stay exact, as do CPU-presented frames.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

/// The pixels a pattern's features must span to draw steadily under
/// temporal antialiasing. Its frames sample each pixel up to half a pixel
/// off its middle, which moves the phase at which a pixel samples a pattern
/// `p` pixels long by up to `1 / (2 p)` of its period from frame to frame:
/// no more than an eighth of it where `p` is 4. Finer patterns draw as
/// their average: waves' slopes as roughness through their variance, foam's
/// bubbles as their expected cover. Shaders mirror it (`STEADY_PIXELS`).
pub const STEADY_PIXELS: f32 = 4.0;

settings! {
    /// Temporal antialiasing settings: whether ordinary GPU-presented frames
    /// jitter and resolve against their history, and how the resolve keeps
    /// moving edges and particles from smearing. Its targets cost 16 bytes
    /// per canvas pixel while it runs. Off by default.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneTemporalAntialiasing")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct TemporalAntialiasing {
        /// Whether ordinary GPU-presented frames resolve temporally.
        pub enabled: bool = false => any,
        /// Whether each pixel follows the motion of the nearest surface in
        /// the 3x3 pixels around it: the edges of nearer surfaces moving over
        /// farther ones bring their history along instead of leaving a trail,
        /// for nine depth reads a pixel.
        pub nearest_motion: bool = true => any,
        /// Whether the history keeps within a box rounded between the range
        /// of the pixel's four edge neighbors and that of all eight: thin and
        /// diagonal features flicker less, for four more reads a pixel.
        pub rounded_clamp: bool = false => any,
        /// How strongly particles' coverage of a pixel favors this frame's
        /// color there: the frame's share is at least this times their
        /// coverage, so smoke and spray do not smear as they move. 0 leaves
        /// them as any surface.
        pub particle_response: f32 = 2.0 => range(0.0, 4.0),
    }
}

impl TemporalAntialiasing {
    /// What a world without these settings reads as: no temporal antialiasing.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
