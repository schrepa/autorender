use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_core::sections::environment_section;
use engine_settings::settings;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneDither")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Dither {
        pub enabled: bool = true => any,
    }
}

environment_section!(Dither, "dither", opt_in: false);

impl Dither {
    pub fn off() -> Self {
        Self { enabled: false }
    }
}
