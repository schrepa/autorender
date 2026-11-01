use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_core::sections::environment_section;
use engine_settings::settings;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneHighlights")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Highlights {
        pub enabled: bool = true => any,
        pub capped: bool = true => any,
        pub sun_size: f32 = 0.27 => range(0.0, 10.0),
        pub cap: f32 = 4.0 => at_least(0.0),
        pub light_radius: f32 = 0.1 => range(0.0, 100.0),
    }
}

environment_section!(Highlights, "highlights", opt_in: false);

impl Highlights {
    pub fn off() -> Self {
        Self {
            enabled: false,
            capped: false,
            ..Self::default()
        }
    }
}
