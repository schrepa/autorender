use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::bricks::{Brick, Bricks};
use crate::sections::environment_section;
use engine_settings::settings;

pub struct Culling;

impl Brick for Culling {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<OcclusionCulling>()
            .settings::<SubpixelCulling>()
            .passes(&["occlusion_test", "opaque_phase_two"]);
    }
}

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

environment_section!(OcclusionCulling, "occlusion_culling", opt_in: false);

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

environment_section!(SubpixelCulling, "subpixel_culling", opt_in: false);

impl SubpixelCulling {
    pub fn off() -> Self {
        Self { enabled: false }
    }
}
