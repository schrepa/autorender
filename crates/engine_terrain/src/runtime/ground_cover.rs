use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::{ExtraRules, InvalidSetting, settings};

settings! {
    extra_rules;
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct Cover {
        pub density: f32 = 90.0 => range(0.0, 400.0),
        pub height: [f32; 2] = [0.12, 0.4] => range(0.01, 4.0),
        pub width: f32 = 0.025 => range(0.001, 0.25),
        pub color: [f32; 3] = [0.13, 0.2, 0.05] => range(0.0, 1.0),
        pub sway: f32 = 0.6 => range(0.0, 1.0),
    }
}

impl ExtraRules for Cover {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.height[0] <= self.height[1] {
            Ok(())
        } else {
            Err(InvalidSetting::new("height", engine_settings::Expected::Sorted))
        }
    }

    fn sanitize_extra(self) -> Self {
        let [low, high] = self.height;
        Self {
            height: [
                engine_core::math::min(low, high),
                engine_core::math::max(low, high),
            ],
            ..self
        }
    }
}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneGroundCover")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct GroundCover {
        pub enabled: bool = true => any,
        pub density: f32 = 1.0 => range(0.0, 4.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_blades: Option<u32> = None => budget,
    }
}

impl GroundCover {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
