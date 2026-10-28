use std::sync::Arc;

use bevy_ecs::prelude::Entity;
use engine_core::{Tile, TileKey};
use glam::{Mat3, Vec3};
use parry3d::shape::HeightField;

use crate::geometry::{AffineShape, Bounds};
use crate::{ColliderShape, CollisionLayers, CollisionWorld, PlacedPart};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Solid {
    pub shape: ColliderShape,
    pub center: Vec3,
    pub axes: Mat3,
    pub min: Vec3,
    pub max: Vec3,
}

impl Solid {
    fn new(affine: &AffineShape) -> Self {
        Self {
            shape: affine.shape,
            center: affine.center,
            axes: affine.basis,
            min: affine.bounds.min,
            max: affine.bounds.max,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacedSolid {
    pub terrain: Entity,
    pub part: PlacedPart,
    pub solid: Solid,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColliderSolid {
    pub entity: Entity,
    pub layers: CollisionLayers,
    pub solid: Solid,
}

pub struct GroundSolid<'a> {
    pub terrain: Entity,
    pub key: TileKey,
    pub tile: &'a Arc<Tile>,
    pub center: Vec3,
    pub min: Vec3,
    pub max: Vec3,
    pub field: &'a Arc<HeightField>,
}

impl CollisionWorld {
    pub fn ground_solids(&self) -> impl Iterator<Item = GroundSolid<'_>> {
        self.terrain.solids()
    }

    pub fn placed_solids_in(&self, min: Vec3, max: Vec3) -> impl Iterator<Item = PlacedSolid> + '_ {
        self.placed.solids_in(Bounds { min, max })
    }

    pub fn collider_solids_in(
        &self,
        min: Vec3,
        max: Vec3,
    ) -> impl Iterator<Item = ColliderSolid> + '_ {
        self.entries_in(Bounds { min, max })
            .filter(|(_, collider, _)| !collider.is_sensor())
            .map(|(entity, collider, affine)| ColliderSolid {
                entity,
                layers: collider.layers(),
                solid: Solid::new(affine),
            })
    }

    pub fn ground_revision(&self) -> Option<u64> {
        self.terrain.revision()
    }
}

impl crate::placed::PlacedShapes {
    pub(crate) fn solids_in(&self, bounds: Bounds) -> impl Iterator<Item = PlacedSolid> + '_ {
        self.objects_in(bounds)
            .map(|(terrain, part, affine)| PlacedSolid {
                terrain,
                part,
                solid: Solid::new(affine),
            })
    }
}
