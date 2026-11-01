use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

pub const MAX_OCCUPANCY_LEVELS: u32 = 8;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneOccupancy")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Occupancy {
        pub enabled: bool = true => any,
        pub radius: f32 = 512.0 => range(64.0, 4096.0),
        pub resolution: u32 = 1024 => multiple(32, 128, 4096),
        pub levels: u32 = 4 => range(1, MAX_OCCUPANCY_LEVELS),
        pub strength: f32 = 0.75 => range(0.0, 1.0),
    }
}

impl Occupancy {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
