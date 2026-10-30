use std::any::Any;
use std::sync::Arc;

use bevy_ecs::prelude::{Entity, Resource, World};
use glam::{Mat4, Vec3};

use crate::smoke::{SmokeEntry, SmokeResult, SmokeSession};
use crate::{Bricks, List, TileKey};

pub struct WorldSolidsList;

impl List for WorldSolidsList {
    const NAME: &'static str = "world_solids";
    const SMOKES: &'static [SmokeEntry] = &[SmokeEntry {
        name: "list_world_solids",
        brick: Self::NAME,
        gpu: false,
        run: world_solids,
    }];
}

pub trait FieldSamples: Send + Sync + 'static {
    fn height(&self, x: i32, z: i32) -> f32;

    fn hole(&self, x: u32, z: u32) -> bool;

    fn same(&self, other: &dyn FieldSamples) -> bool;

    fn as_any(&self) -> &dyn Any;
}

#[derive(Clone)]
pub struct SolidField {
    pub owner: Entity,
    pub key: TileKey,
    pub origin: Vec3,
    pub spacing: f32,
    pub bounds: [f32; 2],
    pub samples: Arc<dyn FieldSamples>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SolidShape {
    Cuboid { half_extents: Vec3 },
    Sphere { radius: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldSolid {
    pub owner: Entity,
    pub layer: u16,
    pub asset: u16,
    pub shape: SolidShape,
    pub model: Mat4,
    pub memberships: u32,
    pub filter: u32,
}

pub trait SolidsFace: Send + Sync + 'static {
    fn revision(&self, world: &World) -> u64;

    fn fields(&self, _world: &World) -> Vec<SolidField> {
        Vec::new()
    }

    fn solids(&self, _world: &World) -> Vec<WorldSolid> {
        Vec::new()
    }
}

pub struct SolidsMember {
    pub brick: &'static str,
    pub name: &'static str,
    face: Box<dyn SolidsFace>,
}

impl SolidsMember {
    pub fn face(&self) -> &dyn SolidsFace {
        &*self.face
    }
}

#[derive(Resource, Default)]
pub struct WorldSolids {
    members: Vec<SolidsMember>,
}

impl WorldSolids {
    pub fn add(&mut self, brick: &'static str, name: &'static str, face: impl SolidsFace) {
        let face: Box<dyn SolidsFace> = Box::new(face);
        match self
            .members
            .iter_mut()
            .find(|member| member.brick == brick && member.name == name)
        {
            Some(member) => member.face = face,
            None => self.members.push(SolidsMember { brick, name, face }),
        }
    }

    pub fn members(&self) -> &[SolidsMember] {
        &self.members
    }

    pub fn revision(&self, world: &World) -> u64 {
        self.members
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |folded, member| {
                (folded ^ member.face.revision(world)).wrapping_mul(0x0100_0000_01b3)
            })
    }

    pub fn fields(&self, world: &World) -> Vec<SolidField> {
        self.members
            .iter()
            .flat_map(|member| member.face.fields(world))
            .collect()
    }

    pub fn solids(&self, world: &World) -> Vec<WorldSolid> {
        self.members
            .iter()
            .flat_map(|member| member.face.solids(world))
            .collect()
    }
}

impl Bricks<'_> {
    pub fn world_solids(&mut self, name: &'static str, face: impl SolidsFace) -> &mut Self {
        let brick = self.name();
        let world = self.world_mut();
        if !world.contains_resource::<WorldSolids>() {
            world.init_resource::<WorldSolids>();
        }
        world.resource_mut::<WorldSolids>().add(brick, name, face);
        self.list::<WorldSolidsList>()
    }
}

fn world_solids(_session: &mut dyn SmokeSession) -> SmokeResult {
    Ok(())
}
