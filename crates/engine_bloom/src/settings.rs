use bevy_ecs::prelude::Resource;
use engine_core::EnvironmentSection;
use engine_settings::{InvalidSetting, settings};
use serde::{Deserialize, Serialize};

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneBloom")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Bloom {
        pub intensity: f32 = 0.6 => range(0.0, 4.0),
        pub threshold: f32 = 2.0 => at_least(0.0),
    }
}

impl Bloom {
    pub fn off() -> Self {
        Self {
            intensity: 0.0,
            ..Self::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.intensity > 0.0
    }
}

impl EnvironmentSection for Bloom {
    const KEY: &'static str = "bloom";
    const NAME: &'static str = "Bloom";
    const OPT_IN: bool = false;

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
        schemars::schema_for!(Bloom)
    }
}
