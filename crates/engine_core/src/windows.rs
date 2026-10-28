use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneWindows")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Windows {
        pub enabled: bool = true => any,
        pub dusk: f32 = -3.0 => range(-90.0, 90.0),
        pub spread: f32 = 12.0 => above_to(0.0, 90.0),
        pub dark: f32 = 0.1 => range(0.0, 1.0),
    }
}

impl Windows {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
