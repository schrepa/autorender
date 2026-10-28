use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneSky")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Sky {
        pub enabled: bool = true => any,
        pub planet_radius: f32 = 6360.0 => range(100.0, 100_000.0),
        pub atmosphere_height: f32 = 100.0 => range(1.0, 1000.0),
        pub rayleigh: [f32; 3] = [5.802e-3, 13.558e-3, 33.1e-3] => range(0.0, 1.0),
        pub rayleigh_height: f32 = 8.0 => range(0.1, 100.0),
        pub mie: f32 = 3.996e-3 => range(0.0, 1.0),
        pub mie_absorption: f32 = 4.4e-3 => range(0.0, 1.0),
        pub mie_height: f32 = 1.2 => range(0.1, 100.0),
        pub mie_g: f32 = 0.8 => range(0.0, 0.99),
        pub ozone: [f32; 3] = [0.65e-3, 1.881e-3, 0.085e-3] => range(0.0, 1.0),
        pub ground_albedo: [f32; 3] = [0.3, 0.3, 0.3] => range(0.0, 1.0),
        pub sun_intensity: f32 = 3.0 => range(0.0, 1000.0),
        pub sun_size: f32 = 0.27 => range(0.01, 10.0),
        pub aerial: f32 = 1.0 => range(0.0, 100.0),
        pub altitude: f32 = 0.0 => range(0.0, 100_000.0),
        pub adaptation: f32 = 0.5 => range(0.0, 1.0),
    }
}

impl Sky {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneHaze")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Haze {
        pub enabled: bool = true => any,
        pub visibility: f32 = 0.0 => at_least(0.0),
        pub strength: f32 = 0.7 => range(0.0, 4.0),
        pub forward: f32 = 0.6 => range(0.0, 0.95),
        pub reach: f32 = 300.0 => range(0.0, 100_000.0),
        pub spread: f32 = 0.6 => range(0.0, 1.0),
        pub size: f32 = 60.0 => range(0.1, 100_000.0),
    }
}

impl Haze {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneNightSky")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct NightSky {
        pub stars: f32 = 1.0 => range(0.0, 8.0),
        pub moon: f32 = 1.0 => range(0.0, 8.0),
        pub moon_age: f32 = 0.5 => range(0.0, 1.0),
        pub cirrus: f32 = 0.35 => range(0.0, 1.0),
        pub star_density: f32 = 1.0 => range(0.0, 2.0),
        pub moon_size: f32 = 1.0 => range(0.25, 4.0),
        pub cirrus_height: f32 = 9000.0 => range(500.0, 50_000.0),
        pub cirrus_drift: [f32; 2] = [6.0, 1.5] => range(-1000.0, 1000.0),
    }
}

impl NightSky {
    pub fn off() -> Self {
        Self {
            stars: 0.0,
            moon: 0.0,
            cirrus: 0.0,
            ..Self::default()
        }
    }
}
