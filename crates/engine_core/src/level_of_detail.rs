use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::bricks::{Brick, Bricks};
use crate::sections::environment_section;
use engine_settings::settings;

impl Brick for LevelOfDetail {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.settings::<LevelOfDetail>();
    }
}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneLevelOfDetail")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct LevelOfDetail {
        pub enabled: bool = true => any,
        pub pixel_error: f32 = 1.0 => range(0.25, 64.0),
        pub impostors: bool = true => any,
        pub impostor_pixels: f32 = 128.0 => range(8.0, 4096.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_impostors: Option<u32> = None => budget,
        pub fade_seconds: f32 = 0.5 => range(0.0, 4.0),
    }
}

environment_section!(LevelOfDetail, "level_of_detail", opt_in: false);

impl LevelOfDetail {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
