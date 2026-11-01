use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_core::sections::environment_section;
use engine_settings::settings;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneAmbientOcclusion")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct AmbientOcclusion {
        pub intensity: f32 = 0.8 => range(0.0, 1.0),
        pub radius: f32 = 2.5 => above(0.0),
    }
}

environment_section!(AmbientOcclusion, "ambient_occlusion", opt_in: false);

impl AmbientOcclusion {
    pub fn off() -> Self {
        Self {
            intensity: 0.0,
            ..Self::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.intensity > 0.0
    }
}
