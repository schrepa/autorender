use engine::prelude::Resource;
use glam::{Mat3, Vec3};

use crate::rules::Team;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Look {
    Projectile,
    Smoke,
    DamageZone,
    SlowZone,
    HealZone,
    SightZone,
    RevealZone,
    Curtain,
    Wall,
    Trap,
    Wire,
    Sensor,
    StrikeMark,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Form {
    Sphere {
        centre: Vec3,
        radius: f32,
    },
    Box {
        centre: Vec3,
        half_extents: Vec3,
        axes: Mat3,
    },
    Segment {
        a: Vec3,
        b: Vec3,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shown {
    pub look: Look,
    pub form: Form,
    pub team: Team,
    pub age_ticks: u32,
    pub lifetime_ticks: Option<u32>,
    pub strength: f32,
    pub armed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CueLook {
    Explosion,
    Flash,
    Pulse,
    StrikeImpact,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cue {
    pub look: CueLook,
    pub at: Vec3,
    pub size: f32,
    pub ticks: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HumanScreen {
    pub blindness: f32,
    pub sight_limit_m: Option<f32>,
    pub damage: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blob {
    pub at: Vec3,
    pub radius: f32,
    pub height: f32,
    pub strength: f32,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct GroundMarks {
    pub blobs: Vec<Blob>,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct EffectsFeed {
    pub tick: u64,
    pub shown: Vec<Shown>,
    pub cues: Vec<Cue>,
    pub human: HumanScreen,
}
