use bevy_ecs::resource::Resource;
use engine_settings::Error;
use serde::Serialize;

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

#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct LevelRequest {
    pub path: Option<String>,
}

#[derive(Resource, Clone, Debug, Default, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum LevelProgress {
    #[default]
    Idle,
    Reading {
        done: u32,
        total: u32,
    },
    Building,
    Failed {
        error: Error,
    },
}
