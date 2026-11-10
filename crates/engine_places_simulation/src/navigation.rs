use bevy_ecs::prelude::{Component, Entity};
use engine_math::clamp;
use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

#[derive(
    Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[require(NavWalk)]
#[serde(default, deny_unknown_fields)]
pub struct NavAgent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub destination: Option<[f32; 2]>,
    pub speed: f32,
    pub turn: f32,
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
    Refused,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavStep {
    pub to: Vec3,
    pub facing: Quat,
    pub blocked: Option<Entity>,
    pub taken: bool,
}

#[derive(Component, Clone, Debug, Default)]
pub struct NavWalk {
    pub status: NavStatus,
    pub destination: Option<[f32; 2]>,
    pub points: Vec<Vec3>,
    pub next: usize,
    pub length: f32,
    pub searched: usize,
    pub step: Option<NavStep>,
}

impl NavWalk {
    pub fn walked(&self, at: Vec2) -> Option<f32> {
        let last = self.points.len().checked_sub(1)?;
        let a = self.points[self.next.saturating_sub(1).min(last)];
        let b = self.points[self.next.min(last)];
        let span = Vec2::new(b.x - a.x, b.z - a.z);
        let along = Vec2::new(at.x - a.x, at.y - a.z);
        let t = if span.length_squared() > 1.0e-8 {
            clamp(along.dot(span) / span.length_squared(), 0.0, 1.0)
        } else {
            1.0
        };
        Some(a.y + (b.y - a.y) * t)
    }
}
