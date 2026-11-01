use std::any::Any;
use std::sync::Arc;

use bevy_ecs::prelude::{Component, Entity};
use glam::{Mat4, Vec3};

use crate::ground::TileKey;

pub trait FieldSamples: Send + Sync + 'static {
    fn height(&self, x: i32, z: i32) -> f32;

    fn hole(&self, x: u32, z: u32) -> bool;

    fn same(&self, other: &dyn FieldSamples) -> bool;

    fn as_any(&self) -> &dyn Any;
}

#[derive(Clone)]
pub struct FieldTile {
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

pub trait SolidField: Send + Sync + 'static {
    fn fields(&self) -> &[FieldTile] {
        &[]
    }

    fn solids(&self) -> &[WorldSolid] {
        &[]
    }
}

#[derive(Component, Clone)]
pub struct SolidSource {
    pub field: Arc<dyn SolidField>,
}

struct NoSolids;

impl SolidField for NoSolids {}

impl Default for SolidSource {
    fn default() -> Self {
        Self {
            field: Arc::new(NoSolids),
        }
    }
}
