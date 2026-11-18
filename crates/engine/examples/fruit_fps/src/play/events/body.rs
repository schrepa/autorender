use engine::prelude::Resource;
use glam::Vec3;
use serde::Serialize;

use crate::solids::Surface;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stride {
    Run,
    Walk,
    CrouchWalk,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Footstep {
    pub slot: u32,
    pub position: Vec3,
    pub surface: Surface,
    pub stride: Stride,
    pub loud: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Jump {
    pub slot: u32,
    pub position: Vec3,
    pub surface: Option<Surface>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Landing {
    pub slot: u32,
    pub position: Vec3,
    pub surface: Surface,
    pub fall_speed: f32,
    pub hard: bool,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct BodyEvents {
    pub tick: u64,
    pub footsteps: Vec<Footstep>,
    pub jumps: Vec<Jump>,
    pub landings: Vec<Landing>,
}
