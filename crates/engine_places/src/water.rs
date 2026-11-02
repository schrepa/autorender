use std::sync::Arc;

use bevy_ecs::prelude::{Component, Entity, Resource};
use engine_heightfield::{HydrologyBake, TileKey};
use engine_place::{Slot, Stamp};
use glam::{Mat3, Vec3};
use serde::Serialize;

#[derive(Clone, Debug, PartialEq)]
pub struct WaterTileErrors {
    pub terrain: Entity,
    pub key: TileKey,
    pub stamp: Stamp,
    pub errors: Arc<[u16]>,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct Waterline {
    pub subdivision_error: f32,
    pub subdivision_leaves: u32,
    pub watered: Vec<Entity>,
    pub errors: Vec<WaterTileErrors>,
}

impl Slot for Waterline {
    const NAME: &'static str = "water";
    const WGSL: Option<&'static str> = Some(include_str!("places/water.wgsl"));
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

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct WaterFlow {
    pub samples: Vec<WaterSample>,
}

impl Slot for WaterFlow {
    const NAME: &'static str = "water_flow";
}

pub trait WaterQuery: Send + Sync + 'static {
    fn water(&self, point: Vec3) -> Option<WaterSample>;

    fn reaches(&self) -> Vec<(Vec3, Vec3)> {
        Vec::new()
    }
}

#[derive(Component, Clone)]
pub struct WaterBody {
    pub precedence: i32,
    pub shape: Arc<dyn WaterQuery>,
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WaterWork {
    pub uploaded: u64,
}

impl Slot for WaterWork {
    const NAME: &'static str = "water_work";
}

#[derive(Resource, Clone, Default)]
pub struct Hydrology {
    pub bakes: Vec<(Entity, Arc<HydrologyBake>)>,
}

impl Slot for Hydrology {
    const NAME: &'static str = "hydrology";
}
