use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneVirtualGeometry")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct VirtualGeometry {
        pub enabled: bool = true => any,
        pub pixel_error: f32 = 1.5 => range(0.25, 64.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_resident_pages: Option<u32> = None => budget,
        #[serde(skip_serializing_if = "is_false")]
        pub visibility_buffer: bool = false => any,
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

impl VirtualGeometry {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
