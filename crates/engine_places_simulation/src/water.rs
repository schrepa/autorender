use std::sync::Arc;

use bevy_ecs::prelude::{Component, Resource};
use engine_place::Slot;
use glam::{Mat3, Vec3};
use serde::Serialize;

use crate::ground::GroundRead;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterObstacle {
    pub center: Vec3,
    pub axes: Mat3,
    pub round: bool,
    pub velocity: Vec3,
}

impl WaterObstacle {
    pub fn reach(&self) -> Vec3 {
        let rows = self.axes.transpose();
        let along = |row: Vec3| {
            if self.round {
                row.length()
            } else {
                row.abs().element_sum()
            }
        };
        Vec3::new(along(rows.x_axis), along(rows.y_axis), along(rows.z_axis))
    }
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct WaterObstacles {
    pub obstacles: Vec<WaterObstacle>,
}

impl Slot for WaterObstacles {
    const NAME: &'static str = "water_obstacles";
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WaterKind {
    River,
    Lake,
    Fall,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, schemars::JsonSchema)]
pub struct WaterSample {
    pub kind: WaterKind,
    pub surface: f32,
    pub depth: f32,
    pub velocity: [f32; 3],
    pub foam: f32,
}

pub trait WaterQuery: Send + Sync + 'static {
    fn water(&self, point: Vec3) -> GroundRead<Option<WaterSample>>;

    fn reaches(&self) -> Vec<(Vec3, Vec3)> {
        Vec::new()
    }
}

#[derive(Component, Clone)]
pub struct WaterBody {
    pub precedence: i32,
    pub shape: Arc<dyn WaterQuery>,
}
