use engine::prelude::Resource;
use glam::Vec3;

use crate::bots::{BotDebug, SquadDebug};

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
}
