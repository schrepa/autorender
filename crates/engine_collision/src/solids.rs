//! The collision world's solids as a solver mirrors them: terrains'
//! heightfields, the objects placed on them (bridges' decks among them) and
//! the colliders that are not sensors, each with what it comes from, its
//! primitive and its pose. A physics solver holding these collides its bodies
//! with exactly what rays, overlaps and shape casts hit.

use std::sync::Arc;

use bevy_ecs::prelude::Entity;
use engine_core::{Tile, TileKey};
use glam::{Mat3, Vec3};
use parry3d::shape::HeightField;

use crate::geometry::{AffineShape, Bounds};
use crate::{ColliderShape, CollisionLayers, CollisionWorld, PlacedPart};

/// A solid as the collision world holds it: its primitive, its center and
/// the axes mapping the unit primitive into the world (columns scaled by its
/// extents, sheared as its transform shears it), and its world bounds.
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

/// A placed object or bridge deck: its terrain, its placement and its solid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacedSolid {
    pub terrain: Entity,
    pub part: PlacedPart,
    pub solid: Solid,
}

/// A collider that is not a sensor: its entity, its layers and its solid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColliderSolid {
    pub entity: Entity,
    pub layers: CollisionLayers,
    pub solid: Solid,
}

/// A resident level-0 terrain tile's heightfield: its terrain, key and the
/// tile it stands for (edits replace tiles under the same key), its world
/// center (heightfields stand centered) and the field, which solvers share:
/// a tile replaced over the same ground keeps it.
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
    /// The terrain heightfields colliding now.
    pub fn ground_solids(&self) -> impl Iterator<Item = GroundSolid<'_>> {
        self.terrain.solids()
    }

    /// The placed objects and bridge decks colliding now whose bounds meet
    /// the box from `min` to `max`.
    pub fn placed_solids_in(&self, min: Vec3, max: Vec3) -> impl Iterator<Item = PlacedSolid> + '_ {
        self.placed.solids_in(Bounds { min, max })
    }

    /// The colliders that are not sensors whose bounds meet the box from
    /// `min` to `max`, as the last refresh admitted them.
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

    /// Advances whenever the terrains' solids may have changed: the
    /// terrains' revision their heightfields and placed objects were last
    /// built at.
    pub fn ground_revision(&self) -> Option<u64> {
        self.terrain.revision()
    }
}

impl crate::placed::PlacedShapes {
    /// The objects whose bounds meet `bounds`, as solvers mirror them.
    pub(crate) fn solids_in(&self, bounds: Bounds) -> impl Iterator<Item = PlacedSolid> + '_ {
        self.objects_in(bounds)
            .map(|(terrain, part, affine)| PlacedSolid {
                terrain,
                part,
                solid: Solid::new(affine),
            })
    }
}
