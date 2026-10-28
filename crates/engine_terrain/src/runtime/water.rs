use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneWater")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Water {
        pub enabled: bool = true => any,
        pub color: [f32; 3] = [0.012, 0.045, 0.05] => range(0.0, 1.0),
        pub clarity: f32 = 2.5 => range(0.1, 100.0),
        pub sediment: f32 = 0.15 => range(0.0, 50.0),
        pub sediment_color: [f32; 3] = [0.05, 0.065, 0.03] => range(0.0, 1.0),
        pub simulation_cell: f32 = 1.0 => range(0.25, 8.0),
        pub live_budget: u32 = 2_097_152 => range(65_536, 67_108_864),
        pub foam_points: u32 = 1024 => range(64, 8192),
        pub subdivision_error: f32 = 1.0 => range(0.25, 64.0),
        pub subdivision_leaves: u32 = 65_536 => range(1024, 4_000_000),
    }
}

impl Water {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
