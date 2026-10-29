#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use bevy_ecs::prelude::*;
use engine_core::{Brick, BrickStats, Bricks, Engine, EnvironmentSection, Error, Presentation};
use engine_settings::{InvalidSetting, settings};
use serde::{Deserialize, Serialize};

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneTemplate")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Template {
        pub enabled: bool = true => any,
    }
}

impl EnvironmentSection for Template {
    const KEY: &'static str = "template";
    const NAME: &'static str = "Template";
    const OPT_IN: bool = true;

    fn section_off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    fn section_check(&self) -> Result<(), InvalidSetting> {
        self.validate()
    }

    fn section_sanitized(self) -> Self {
        self.sanitized()
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(Template)
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TemplateStats {
    pub frames: u32,
}

impl BrickStats for TemplateStats {
    const KEY: &'static str = "template";

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(TemplateStats)
    }
}

fn count_frames(settings: Option<Res<Template>>, mut stats: ResMut<TemplateStats>) {
    if settings.is_some_and(|settings| settings.enabled) {
        stats.frames = stats.frames.wrapping_add(1);
    }
}

pub fn smoke(_: &mut Engine) -> Result<(), Error> {
    Ok(())
}

pub struct TemplateBrick;

impl Brick for TemplateBrick {
    fn register(bricks: &mut Bricks) {
        bricks
            .settings::<Template>()
            .stats::<TemplateStats>()
            .systems(Presentation, count_frames);
    }
}
