use bevy_ecs::prelude::Component;
use serde::{Deserialize, Serialize};

pub use engine_place::{LevelProgress, LevelRequest, Screens};

#[derive(
    Component, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[schemars(rename = "SceneScreen")]
#[serde(deny_unknown_fields)]
pub struct Screen {
    pub name: String,
    #[serde(default)]
    pub pauses: bool,
}
