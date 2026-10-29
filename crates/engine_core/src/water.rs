use bevy_ecs::prelude::*;
use glam::{Mat3, Vec3};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum WaterKind {
    River,
    Lake,
    Fall,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct WaterSample {
    pub kind: WaterKind,
    pub surface: f32,
    pub depth: f32,
    pub velocity: [f32; 3],
    pub foam: f32,
}

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

#[derive(Resource, Default, Debug)]
pub struct WaterObstacles {
    pub obstacles: Vec<WaterObstacle>,
}
