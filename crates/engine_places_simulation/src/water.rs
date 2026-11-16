use std::sync::Arc;

use bevy_ecs::prelude::{Component, Resource};
use engine_place::Slot;
use glam::{Mat3, Vec3};

use crate::ground::GroundRead;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterObstacle {
    pub center: Vec3,
    pub axes: Mat3,
    pub round: bool,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct WaterObstacles {
    pub obstacles: Vec<WaterObstacle>,
}

impl Slot for WaterObstacles {
    const NAME: &'static str = "water_obstacles";
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSample {
    pub surface: f32,
    pub depth: f32,
    pub velocity: [f32; 3],
    pub falling: bool,
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
