use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

use crate::sections::environment_section;
use engine_settings::settings;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneClouds")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Clouds {
        pub enabled: bool = true => any,
        pub coverage: f32 = 0.45 => range(0.0, 1.0),
        pub base: f32 = 1800.0 => range(-10_000.0, 30_000.0),
        pub thickness: f32 = 1400.0 => range(50.0, 10_000.0),
        pub density: f32 = 1.0 => range(0.01, 4.0),
        pub wind: [f32; 2] = [8.0, 3.0] => range(-100.0, 100.0),
        pub scale: f32 = 3000.0 => range(200.0, 50_000.0),
        pub seed: u32 = 1 => any,
        pub shadows: f32 = 0.75 => range(0.0, 1.0),
        pub resolution: u32 = 2048 => multiple(64, 256, 4096),
        pub slices: u32 = 64 => range(1, 64),
        pub in_view: bool = true => any,
    }
}

environment_section!(Clouds, "clouds", opt_in: true);

impl Clouds {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
