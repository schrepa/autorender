use bevy_ecs::prelude::Resource;
use engine_core::{BrickStats, EnvironmentSection};
use engine_settings::{InvalidSetting, settings};
use engine_terrain::hydraulics::GRAVITY;
use serde::{Deserialize, Serialize};

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "ScenePhysics")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Physics {
        pub enabled: bool = true => any,
        pub gravity: [f32; 3] = [0.0, -GRAVITY, 0.0] => range(-100.0, 100.0),
        pub water_density: f32 = 1000.0 => range(0.0, 20_000.0),
        pub air_density: f32 = 1.225 => range(0.0, 100.0),
        pub iterations: u32 = 4 => range(1, 16),
    }
}

impl Physics {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

impl EnvironmentSection for Physics {
    const KEY: &'static str = "physics";
    const NAME: &'static str = "Physics";
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
        schemars::schema_for!(Physics)
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PhysicsStats {
    pub bodies: u32,
    pub awake: u32,
    pub held: u32,
    pub wet: u32,
    pub solids: u32,
    pub contacts: u32,
    pub water_samples: u32,
}

impl BrickStats for PhysicsStats {
    const KEY: &'static str = "physics";

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(PhysicsStats)
    }
}
