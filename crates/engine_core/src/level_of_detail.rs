use bevy_ecs::prelude::{Resource, World};
use serde::{Deserialize, Serialize};

use crate::bricks::{
    Brick, Bricks, DetailConsumer, DetailConsumersList, DrawState, ExplainRecord, ExplainRequest,
    Join,
};
use crate::sections::environment_section;
use engine_settings::settings;

impl Brick for LevelOfDetail {
    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<LevelOfDetail>()
            .join(Join::<DetailConsumersList>::new().cpu(DetailConsumer {
                name: "impostors",
                setting: "level_of_detail.max_impostors",
                stat: "level_of_detail.impostor_cut",
            }))
            .explain(explain_level);
    }
}

const IMPOSTOR_LEVEL: u32 = 3;

fn explain_level(world: &World, request: &ExplainRequest) -> Option<ExplainRecord> {
    let ExplainRequest::Entity(entity) = request else {
        return None;
    };
    let DrawState::Level(level) = entity.drawn else {
        return None;
    };
    let settings = crate::section_in::<LevelOfDetail>(world);
    Some(
        ExplainRecord::new("level")
            .with("entity", &entity.id)
            .with("level", level)
            .with("impostor", level == IMPOSTOR_LEVEL)
            .with("pixel_error", settings.pixel_error)
            .with("impostor_pixels", settings.impostor_pixels),
    )
}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneLevelOfDetail")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct LevelOfDetail {
        pub enabled: bool = true => any,
        pub pixel_error: f32 = 1.0 => range(0.25, 64.0),
        pub impostors: bool = true => any,
        pub impostor_pixels: f32 = 128.0 => range(8.0, 4096.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_impostors: Option<u32> = None => budget,
        pub fade_seconds: f32 = 0.5 => range(0.0, 4.0),
    }
}

environment_section!(LevelOfDetail, "level_of_detail", opt_in: false);

impl LevelOfDetail {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
