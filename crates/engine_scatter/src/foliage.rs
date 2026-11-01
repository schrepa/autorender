use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_core::sections::environment_section;
use engine_settings::settings;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneFoliage")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Foliage {
        pub transmission: f32 = 1.0 => range(0.0, 4.0),
        pub variation: f32 = 1.0 => range(0.0, 1.0),
        pub hashed_cutoff: bool = false => any,
        pub ground_blend: f32 = 0.4 => range(0.0, 4.0),
        pub crown_depth: f32 = 2.0 => range(0.0, 8.0),
    }
}

environment_section!(Foliage, "foliage", opt_in: false);

impl Foliage {
    pub fn off() -> Self {
        Self {
            transmission: 0.0,
            variation: 0.0,
            hashed_cutoff: false,
            ground_blend: 0.0,
            crown_depth: 0.0,
        }
    }

    pub fn any(&self) -> bool {
        self.transmission > 0.0
            || self.variation > 0.0
            || self.hashed_cutoff
            || self.ground_blend > 0.0
            || self.crown_depth > 0.0
    }
}
