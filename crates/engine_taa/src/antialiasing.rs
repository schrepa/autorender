use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_core::sections::environment_section;
use engine_settings::settings;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneTemporalAntialiasing")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct TemporalAntialiasing {
        pub enabled: bool = false => any,
        pub nearest_motion: bool = true => any,
        pub rounded_clamp: bool = false => any,
        pub particle_response: f32 = 2.0 => range(0.0, 4.0),
    }
}

environment_section!(TemporalAntialiasing, "temporal_antialiasing", opt_in: false);

impl TemporalAntialiasing {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
