use bevy_ecs::prelude::*;
use glam::{Mat3, Vec3};
use serde::Serialize;

use crate::Bricks;

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

pub trait WaterFace: Send + Sync + 'static {
    fn water(&self, world: &World, point: Vec3) -> Option<WaterSample>;
}

#[derive(Resource, Default)]
pub struct WaterBodies {
    bodies: Vec<(&'static str, Box<dyn WaterFace>)>,
}

impl WaterBodies {
    pub fn add(&mut self, brick: &'static str, face: impl WaterFace) {
        let face: Box<dyn WaterFace> = Box::new(face);
        match self.bodies.iter_mut().find(|(known, _)| *known == brick) {
            Some(body) => body.1 = face,
            None => self.bodies.push((brick, face)),
        }
    }

    pub fn water(&self, world: &World, point: Vec3) -> Option<WaterSample> {
        self.bodies.iter().find_map(|(_, face)| face.water(world, point))
    }

    pub fn water_in(world: &World, point: Vec3) -> Option<WaterSample> {
        world.get_resource::<Self>()?.water(world, point)
    }
}

impl Bricks<'_> {
    pub fn water_body(&mut self, face: impl WaterFace) -> &mut Self {
        let brick = self.name();
        let world = self.world_mut();
        if !world.contains_resource::<WaterBodies>() {
            world.init_resource::<WaterBodies>();
        }
        world.resource_mut::<WaterBodies>().add(brick, face);
        self
    }
}
