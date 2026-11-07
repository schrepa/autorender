use bevy_ecs::prelude::{Component, Resource};
use engine_place::{ApplyAt, Ask, Slot};
use serde::{Deserialize, Serialize};

pub use engine_place::{Level, LevelView};

#[derive(Resource, Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Screens {
    stack: Vec<String>,
}

impl Screens {
    pub fn go(&mut self, name: impl Into<String>) {
        self.stack.clear();
        self.stack.push(name.into());
    }

    pub fn push(&mut self, name: impl Into<String>) {
        self.stack.push(name.into());
    }

    pub fn pop(&mut self) -> Option<String> {
        self.stack.pop()
    }

    pub fn top(&self) -> Option<&str> {
        self.stack.last().map(String::as_str)
    }

    pub fn stack(&self) -> &[String] {
        &self.stack
    }
}

impl Slot for Screens {
    const NAME: &'static str = "screens";
}

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
