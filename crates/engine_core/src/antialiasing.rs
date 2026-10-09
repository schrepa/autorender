//! Temporal antialiasing: ordinary GPU-presented frames rasterize with a
//! sub-pixel jitter and blend with the previous frame's output, reprojected
//! by per-pixel motion. Development steps, renders, previews and captures
//! stay exact, as do CPU-presented frames.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::environment::InvalidSetting;

/// Temporal antialiasing settings: whether ordinary GPU-presented frames
/// jitter and resolve against their history. Its targets cost 16 bytes per
/// canvas pixel while it runs. Off by default. Requires version 9.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneTemporalAntialiasing")
)]
#[serde(default, deny_unknown_fields)]
pub struct TemporalAntialiasing {
    pub enabled: bool,
}

impl TemporalAntialiasing {
    /// Every value is valid.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        Ok(())
    }

    pub fn sanitized(self) -> Self {
        self
    }
}
