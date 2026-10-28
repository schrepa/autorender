use engine::prelude::*;
use engine::settings::{ExtraRules, InvalidSetting, Setting, settings};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum StressScene {
    #[default]
    Field,
    City,
    Metro,
    Dense,
    Highlands,
}

impl StressScene {
    pub const ALL: [Self; 5] = [
        Self::Field,
        Self::City,
        Self::Metro,
        Self::Dense,
        Self::Highlands,
    ];

    pub fn name(self) -> &'static str {
        self.world().name()
    }

    pub fn label(self) -> &'static str {
        self.world().label()
    }
}

impl Setting for StressScene {}

pub const MESH_VARIETIES: [u32; 4] = [1, 8, 32, 64];

settings! {
    extra_rules;
    #[derive(Resource, Clone, Debug, Deserialize, Serialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct StressConfig {
        pub seed: u32 = 1337 => range(0, u32::MAX),
        pub renderables: u32 = 10_000 => range(1, 4_000_000),
        pub moving_percent: u32 = 25 => range(0, 100),
        pub hierarchy: bool = true => any,
        pub interpolation: bool = true => any,
        pub colliders: u32 = 1_000 => range(0, 10_000),
        pub sensor_percent: u32 = 10 => range(0, 100),
        pub dense_colliders: bool = false => any,
        pub queries_per_tick: u32 = 32 => range(0, 512),
        pub bodies: u32 = 0 => range(0, 64),
        #[cfg_attr(feature = "schema", schemars(transform = mesh_varieties))]
        pub mesh_variety: u32 = 8 => any,
        pub textured: bool = true => any,
        pub culling: bool = true => any,
        pub scene: StressScene = StressScene::Field => any,
    }
}

impl ExtraRules for StressConfig {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if !MESH_VARIETIES.contains(&self.mesh_variety) {
            return Err(InvalidSetting::new(
                "mesh_variety",
                format!("must be one of {MESH_VARIETIES:?}"),
            ));
        }
        if self.colliders > self.renderables {
            return Err(InvalidSetting::new(
                "colliders",
                "must be at most renderables",
            ));
        }
        Ok(())
    }

    fn sanitize_extra(mut self) -> Self {
        if !MESH_VARIETIES.contains(&self.mesh_variety) {
            self.mesh_variety = Self::default().mesh_variety;
        }
        self.colliders = self.colliders.min(self.renderables);
        self
    }
}

#[cfg(feature = "schema")]
fn mesh_varieties(schema: &mut schemars::Schema) {
    schema.insert("enum".into(), serde_json::json!(MESH_VARIETIES));
}

impl StressConfig {
    pub fn from_json(source: &str) -> engine::Result<Self> {
        if source.len() > 4096 {
            return Err(engine::Error::game_rule("configuration exceeds 4096 bytes"));
        }
        let config: Self = engine::scene::parse_json(source)?;
        config.validate()?;
        Ok(config)
    }
}
