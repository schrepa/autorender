use bevy_ecs::prelude::Resource;
use engine_core::EnvironmentSection;
use engine_settings::{InvalidSetting, settings};
use serde::{Deserialize, Serialize};

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneAutoExposure")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct AutoExposure {
        pub enabled: bool = true => any,
        pub strength: f32 = 1.0 => range(0.0, 1.0),
        pub compensation: f32 = 0.0 => range(-8.0, 8.0),
        pub night: f32 = 1.0 => range(0.0, 16.0),
        pub brighten: f32 = 3.0 => range(0.0, 16.0),
        pub darken: f32 = 1.0 => range(0.0, 16.0),
        pub to_brighter: f32 = 0.5 => range(0.0, 60.0),
        pub to_darker: f32 = 2.0 => range(0.0, 60.0),
        pub skip_dark: f32 = 0.3 => range(0.0, 0.45),
        pub skip_bright: f32 = 0.05 => range(0.0, 0.45),
        pub center: f32 = 2.0 => range(1.0, 16.0),
        pub sky: f32 = 0.5 => range(0.0, 1.0),
    }
}

impl AutoExposure {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

impl EnvironmentSection for AutoExposure {
    const KEY: &'static str = "auto_exposure";
    const NAME: &'static str = "AutoExposure";
    const OPT_IN: bool = true;

    fn section_off() -> Self {
        Self::off()
    }

    fn section_check(&self) -> Result<(), InvalidSetting> {
        self.validate()
    }

    fn section_sanitized(self) -> Self {
        self.sanitized()
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(AutoExposure)
    }
}
