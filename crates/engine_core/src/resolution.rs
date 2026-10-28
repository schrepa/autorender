use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::environment::InvalidSetting;
use engine_settings::{ExtraRules, settings};

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneResolution")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Resolution {
        pub scale: f32 = 1.0 => range(0.25, 2.0),
    }
}

impl Resolution {
    pub fn off() -> Self {
        Self::default()
    }
}

settings! {
    extra_rules;
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneDynamicResolution")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct DynamicResolution {
        pub enabled: bool = true => any,
        pub min_scale: f32 = 0.5 => range(0.25, 1.0),
        pub max_scale: f32 = 1.0 => range(0.25, 1.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub budget_ms: Option<f32> = None => range(1.0, 1000.0),
    }
}

impl ExtraRules for DynamicResolution {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.min_scale > self.max_scale {
            return Err(InvalidSetting::new(
                "min_scale",
                "must be at most max_scale",
            ));
        }
        Ok(())
    }

    fn sanitize_extra(self) -> Self {
        Self {
            min_scale: crate::math::min(self.min_scale, self.max_scale),
            ..self
        }
    }
}

impl DynamicResolution {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
