use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneSkyOcclusion")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct SkyOcclusion {
        pub enabled: bool = true => any,
        pub strength: f32 = 1.0 => range(0.0, 1.0),
        pub floor: f32 = 0.0 => range(0.0, 1.0),
        pub near_distance: f32 = 48.0 => range(8.0, 256.0),
        pub far_distance: f32 = 400.0 => range(64.0, 4000.0),
        pub directions: u32 = 13 => range(5.0, 13.0),
    }
}

impl SkyOcclusion {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    pub fn darkens(&self) -> bool {
        self.enabled && self.strength > 0.0
    }

    pub fn direction_count(&self) -> u32 {
        match self.directions {
            ..9 => 5,
            9..13 => 9,
            _ => 13,
        }
    }
}
