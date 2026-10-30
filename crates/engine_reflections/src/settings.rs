use bevy_ecs::prelude::Resource;
use engine_core::EnvironmentSection;
use engine_settings::{InvalidSetting, settings};
use serde::{Deserialize, Serialize};

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneReflections")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Reflections {
        pub intensity: f32 = 1.0 => range(0.0, 1.0),
        pub max_roughness: f32 = 0.75 => range(0.05, 1.0),
        pub distance: f32 = 120.0 => range(0.001, 100_000.0),
    }
}

impl Reflections {
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

impl EnvironmentSection for Reflections {
    const KEY: &'static str = "reflections";
    const NAME: &'static str = "Reflections";
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
        schemars::schema_for!(Reflections)
    }
}
