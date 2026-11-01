use bevy_ecs::prelude::Component;
use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::ground::HeightHeld;

#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[require(NavWalk, HeightHeld)]
#[serde(default, deny_unknown_fields)]
pub struct NavAgent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub destination: Option<[f32; 2]>,
    pub speed: f32,
    pub turn: f32,
    pub offset: f32,
    pub radius: f32,
    pub max_slope: f32,
    pub max_water: f32,
    pub level: u32,
}

impl Default for NavAgent {
    fn default() -> Self {
        Self {
            destination: None,
            speed: 1.4,
            turn: 360.0,
            offset: 0.0,
            radius: 0.4,
            max_slope: 0.8,
            max_water: 0.6,
            level: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NavStatus {
    #[default]
    Idle,
    Searching,
    Walking,
    Arrived,
    Unreachable,
}

#[derive(Component, Clone, Debug, Default)]
pub struct NavWalk {
    pub status: NavStatus,
    pub destination: Option<[f32; 2]>,
    pub points: Vec<Vec3>,
    pub next: usize,
    pub length: f32,
    pub searched: usize,
}
