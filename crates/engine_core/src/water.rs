use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_settings::Error;
use glam::{Mat3, Vec3};
use serde::Serialize;

use crate::smoke::SmokeEntry;
use crate::{Bricks, GroundSource, List, TileKey};

pub struct WaterBodiesList;

impl List for WaterBodiesList {
    const NAME: &'static str = "water_bodies";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "list_water_bodies",
        brick: Self::NAME,
        gpu: false,
        run: crate::place_smokes::water_bodies,
    }];
}

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

    fn reaches(&self, _world: &World) -> Vec<(Vec3, Vec3)> {
        Vec::new()
    }

    fn surfaces(&self, _world: &World, _ground: &GroundSource) -> Option<Arc<dyn WaterSurfaces>> {
        None
    }
}

pub trait WaterSurfaces: Send + Sync + 'static {
    fn decode(&self, key: TileKey, ground: &[u8]) -> Result<Option<Box<[f32]>>, Error>;
}

pub trait WaterWorker {
    fn surfaces(&self, identity: &[u8], key: TileKey) -> Result<Option<Box<[f32]>>, Error>;
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
        self.bodies
            .iter()
            .find_map(|(_, face)| face.water(world, point))
    }

    pub fn water_in(world: &World, point: Vec3) -> Option<WaterSample> {
        world.get_resource::<Self>()?.water(world, point)
    }

    pub fn reaches(&self, world: &World) -> Vec<(Vec3, Vec3)> {
        self.bodies
            .iter()
            .flat_map(|(_, face)| face.reaches(world))
            .collect()
    }

    pub fn surfaces(&self, world: &World, ground: &GroundSource) -> Vec<Arc<dyn WaterSurfaces>> {
        self.bodies
            .iter()
            .filter_map(|(_, face)| face.surfaces(world, ground))
            .collect()
    }

    pub fn bodies(&self) -> impl ExactSizeIterator<Item = (&'static str, &dyn WaterFace)> + '_ {
        self.bodies.iter().map(|(brick, face)| (*brick, &**face))
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
        self.list::<WaterBodiesList>()
    }
}
