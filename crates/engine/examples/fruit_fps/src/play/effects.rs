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
    pub effect: u32,
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PropPlace {
    Thrown {
        centre: Vec3,
        velocity_mps: Vec3,
        resting: bool,
    },
    Placed {
        contact: Vec3,
        normal: Vec3,
        facing: Vec3,
    },
    Wall {
        base: Vec3,
        axes: Mat3,
        half_extents: Vec3,
    },
    Wire {
        a: Vec3,
        b: Vec3,
        normal_a: Vec3,
        normal_b: Vec3,
    },
    Drop {
        target: Vec3,
        height_m: f32,
        fall_starts_tick: u64,
        impact_tick: u64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct PropShown {
    pub effect: u32,
    pub prop: String,
    pub team: Team,
    pub born_tick: u64,
    pub place: PropPlace,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct EffectsFeed {
    pub tick: u64,
    pub shown: Vec<Shown>,
    pub cues: Vec<Cue>,
    pub human: HumanScreen,
    pub props: Vec<PropShown>,
}
