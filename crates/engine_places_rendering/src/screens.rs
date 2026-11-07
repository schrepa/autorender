use bevy_ecs::prelude::Component;
use engine_place::{ApplyAt, Ask};
use serde::{Deserialize, Serialize};

pub use engine_place::{Level, LevelView, Screens};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScreenAsk {
    Go(String),
    Push(String),
    Pop,
}

impl Ask for ScreenAsk {
    const NAME: &'static str = "screen_change";
    const AT: ApplyAt = ApplyAt::FrameStart;
}

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
