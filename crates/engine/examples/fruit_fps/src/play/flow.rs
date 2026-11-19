use engine::prelude::Resource;

use crate::rules::{Control, Team};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stage {
    #[default]
    Title,
    Select,
    Loading,
    Playing,
    Paused,
    MatchEnd,
}

impl Stage {
    pub fn key(self) -> &'static str {
        match self {
            Stage::Title => "title",
            Stage::Select => "select",
            Stage::Loading => "loading",
            Stage::Playing => "playing",
            Stage::Paused => "paused",
            Stage::MatchEnd => "match_end",
        }
    }

    pub fn lobby(self) -> bool {
        match self {
            Stage::Title | Stage::Select => true,
            Stage::Loading | Stage::Playing | Stage::Paused | Stage::MatchEnd => false,
        }
    }

    pub fn moving(self) -> bool {
        self == Stage::Playing
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeatChoice {
    pub team: Team,
    pub character: String,
    pub control: Control,
    pub level: String,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct Flow {
    pub stage: Stage,
    pub settings_open: bool,
    pub character: String,
    pub level: String,
    pub roster: Vec<SeatChoice>,
    pub game: u64,
    pub progress: f32,
    pub restored: bool,
    pub next_match_left: f32,
}

impl Flow {
    pub fn holds_pointer(&self) -> bool {
        self.settings_open || !self.stage.moving()
    }
}
