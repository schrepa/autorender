use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_core::sections::environment_section;
use engine_settings::settings;

pub const MAX_SHADOW_LEVELS: u32 = 20;
pub const MAX_MOVING_TILES: u32 = 256;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneShadows")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Shadows {
        pub enabled: bool = true => any,
        pub levels: u32 = 16 => range(1, MAX_SHADOW_LEVELS),
        pub finest_texel: f32 = 0.04 => range(0.005, 1.0),
        pub tile_texels: u32 = 128 => power_of_two(64, 512),
        pub atlas_slots: u32 = 1024 => range(64, 4096),
        pub redraw_budget: u32 = 32 => range(1, 4096),
        pub moving_tiles: u32 = MAX_MOVING_TILES => range(0, MAX_MOVING_TILES),
    }
}

environment_section!(Shadows, "shadows", opt_in: false);

impl Shadows {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
