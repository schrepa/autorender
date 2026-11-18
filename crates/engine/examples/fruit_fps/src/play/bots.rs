use engine::prelude::Resource;
use glam::Vec3;

use crate::bots::{BotDebug, Line, SquadDebug, ThrowCounts, ThrowState};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WishSummary {
    pub direction: Vec3,
    pub walk: bool,
    pub fire: bool,
    pub yaw_radians: f32,
    pub pitch_radians: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BotMind {
    pub slot: u32,
    pub mind: BotDebug,
    pub throws: ThrowCounts,
    pub line_up: Option<(Line, ThrowState)>,
    pub goal: Vec3,
    pub wish: WishSummary,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct BotMinds {
    pub tick: u64,
    pub frozen: bool,
    pub problem: Option<String>,
    pub bots: Vec<BotMind>,
    pub squads: [Option<SquadDebug>; 2],
    pub queries: u32,
    pub expanded: u32,
    pub drops: u32,
    pub gifts: u32,
    pub buy_calls: [Option<&'static str>; 2],
    pub sight: TeamSight,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TeamSight {
    pub seen: [Vec<(u32, u32)>; 2],
    pub last: [Vec<(u32, u64, Vec3)>; 2],
}
