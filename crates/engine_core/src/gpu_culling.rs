use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneOcclusionCulling")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct OcclusionCulling {
        pub enabled: bool = true => any,
    }
}

impl OcclusionCulling {
    pub fn off() -> Self {
        Self { enabled: false }
    }
}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneSubpixelCulling")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct SubpixelCulling {
        pub enabled: bool = true => any,
    }
}

impl SubpixelCulling {
    pub fn off() -> Self {
        Self { enabled: false }
    }
}
