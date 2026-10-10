//! Temporal antialiasing: ordinary GPU-presented frames rasterize with a
//! sub-pixel jitter and blend with the previous frame's output, reprojected
//! by per-pixel motion. Development steps, renders, previews and captures
//! stay exact, as do CPU-presented frames.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    /// Temporal antialiasing settings: whether ordinary GPU-presented frames
    /// jitter and resolve against their history. Its targets cost 16 bytes per
    /// canvas pixel while it runs. Off by default. Requires version 9.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneTemporalAntialiasing")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct TemporalAntialiasing {
        /// Whether ordinary GPU-presented frames resolve temporally.
        pub enabled: bool = false => any,
    }
}

impl TemporalAntialiasing {
    /// What a world without these settings reads as: no temporal antialiasing.
    pub fn off() -> Self {
        Self { enabled: false }
    }
}
