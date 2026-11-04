use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::collections::HashMap;
use engine_core::{GlobalTransforms, Transform, TransformsSince, valid_global_matrix};
use engine_places::physics::{
    Collider, ColliderShape, Collision, CollisionLayers, MAX_COLLIDER_EXTENT,
    MAX_COLLISION_COORDINATE, MIN_COLLIDER_EXTENT, PlacedPart,
};
use engine_places::solids::{Bounds, Bvh, FieldTile, SolidField, SolidSource};
use glam::{Mat3, Mat4, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Solid {
    pub shape: ColliderShape,
    pub center: Vec3,
    pub axes: Mat3,
    pub min: Vec3,
    pub max: Vec3,
}

impl Solid {
    fn new(shape: ColliderShape, matrix: Mat4) -> Option<Self> {
        if !placeable(shape, matrix) {
            return None;
        }
        let axes = Mat3::from_mat4(matrix) * Mat3::from_diagonal(shape.dimensions());
        let center = matrix.w_axis.truncate();
        let bounds = Bounds::of_shape(shape, center, axes);
        Some(Self {
            shape,
            center,
            axes,
            min: bounds.min,
            max: bounds.max,
        })
    }

    fn bounds(&self) -> Bounds {
        Bounds {
            min: self.min,
            max: self.max,
        }
    }
}

fn placeable(shape: ColliderShape, matrix: Mat4) -> bool {
    if shape.validate().is_err() {
        return false;
    }
    let position = matrix.w_axis.truncate();
    if !valid_global_matrix(matrix)
        || !(position.is_finite() && position.abs().max_element() <= MAX_COLLISION_COORDINATE)
    {
        return false;
    }
    let basis = Mat3::from_mat4(matrix) * Mat3::from_diagonal(shape.dimensions());
    let lengths = Vec3::new(
        basis.x_axis.length(),
        basis.y_axis.length(),
        basis.z_axis.length(),
    );
    if !lengths.is_finite()
        || lengths.min_element() < MIN_COLLIDER_EXTENT
        || lengths.max_element() > MAX_COLLIDER_EXTENT
    {
        return false;
    }
    let normalized = basis / lengths.max_element();
    let inverse = normalized.inverse();
    let row_sums = inverse.x_axis.abs() + inverse.y_axis.abs() + inverse.z_axis.abs();
    inverse.is_finite() && row_sums.max_element() <= 1000.0
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PlacedSolid {
    pub terrain: Entity,
    pub part: PlacedPart,
    pub solid: Solid,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ColliderSolid {
    pub entity: Entity,
    pub layers: CollisionLayers,
    pub solid: Solid,
}

#[derive(Clone, Copy)]
struct Entry {
    sensor: bool,
    found: ColliderSolid,
}

type ColliderData = (Entity, &'static Collider, Has<Transform>);

fn read(
    globals: &GlobalTransforms,
    (entity, collider, local): (Entity, &Collider, bool),
) -> Option<Solid> {
    if !local {
        return None;
    }
    let matrix = globals.get(entity)?.matrix()?;
    Solid::new(collider.shape(), matrix)
}

#[derive(Resource, Default)]
pub(crate) struct Solids {
    owners: Vec<Entity>,
    sources: Vec<Arc<SolidField>>,
    source_revision: u64,
    entries: Vec<Entry>,
    held: HashMap<Entity, Option<u32>>,
    bounds: Vec<Bounds>,
    bvh: Bvh,
    count: usize,
    limit: Option<usize>,
    moved: u64,
}

impl Solids {
    pub fn collides(&self) -> bool {
        self.limit.is_some()
    }

    pub fn ground(&self) -> impl Iterator<Item = &Arc<FieldTile>> {
        self.sources.iter().flat_map(|source| &source.fields)
    }

    pub fn ground_revision(&self) -> u64 {
        self.source_revision
    }

    pub fn placed_in(&self, min: Vec3, max: Vec3) -> impl Iterator<Item = PlacedSolid> + '_ {
        let bounds = Bounds { min, max };
        self.sources
            .iter()
            .flat_map(move |source| source.solids.query(move |node| node.overlaps(bounds)))
            .filter_map(move |solid| {
                let found = Solid::new(solid.shape, solid.model)?;
                found.bounds().overlaps(bounds).then_some(PlacedSolid {
                    terrain: solid.owner,
                    part: PlacedPart {
                        layer: solid.layer,
                        asset: solid.asset,
                    },
                    solid: found,
                })
            })
    }

    pub fn colliders_in(&self, min: Vec3, max: Vec3) -> impl Iterator<Item = ColliderSolid> + '_ {
        let bounds = Bounds { min, max };
        self.bvh
            .query(move |node| node.overlaps(bounds))
            .map(|index| self.entries[index])
            .filter(move |entry| entry.found.solid.bounds().overlaps(bounds))
            .filter(|entry| !entry.sensor)
            .map(|entry| entry.found)
    }

    fn sync_all(
        &mut self,
        limit: usize,
        colliders: &Query<'_, '_, ColliderData>,
        globals: &GlobalTransforms,
    ) {
        let held = std::mem::take(&mut self.held);
        self.limit = Some(limit);
        self.count = colliders.iter().len();
        self.entries.clear();
        self.bounds.clear();
        self.held.reserve(self.count.min(limit));
        let mut same_leaves = true;
        for data in colliders.iter().take(limit) {
            let (entity, collider, _) = data;
            let slot = read(globals, data).map(|solid| {
                self.bounds.push(solid.bounds());
                self.entries.push(Entry {
                    sensor: collider.is_sensor(),
                    found: ColliderSolid {
                        entity,
                        layers: collider.layers(),
                        solid,
                    },
                });
                (self.entries.len() - 1) as u32
            });
            same_leaves &= held.get(&entity) == Some(&slot);
            self.held.insert(entity, slot);
        }
        same_leaves &= held.len() == self.held.len();
        self.bvh.update(&self.bounds, same_leaves);
    }

    fn sync_moved(
        &mut self,
        moved: &[Entity],
        colliders: &Query<'_, '_, ColliderData>,
        globals: &GlobalTransforms,
    ) -> bool {
        for &entity in moved {
            let Some(&slot) = self.held.get(&entity) else {
                continue;
            };
            let Ok(data) = colliders.get(entity) else {
                return false;
            };
            match (slot, read(globals, data)) {
                (Some(slot), Some(solid)) => {
                    let slot = slot as usize;
                    self.entries[slot].found.solid = solid;
                    self.bounds[slot] = solid.bounds();
                    self.bvh.move_leaf(slot, solid.bounds());
                }
                (None, None) => {}
                (Some(_), None) | (None, Some(_)) => return false,
            }
        }
        if self.bvh.degraded() {
            self.bvh.rebuild(&self.bounds);
        }
        true
    }
}

pub(crate) fn sync_sources(
    sources: Query<(Entity, Ref<SolidSource>)>,
    mut solids: ResMut<Solids>,
) {
    let moved = sources.iter().any(|(_, source)| source.is_changed())
        || !sources
            .iter()
            .map(|(entity, _)| entity)
            .eq(solids.owners.iter().copied());
    if !moved {
        return;
    }
    let solids = &mut *solids;
    solids.owners.clear();
    solids.sources.clear();
    for (entity, source) in &sources {
        solids.owners.push(entity);
        solids.sources.push(Arc::clone(&source.field));
    }
    solids.source_revision = solids.source_revision.wrapping_add(1);
}

pub(crate) fn sync_colliders(
    collision: Res<Collision>,
    colliders: Query<ColliderData>,
    edited: Query<(), Changed<Collider>>,
    globals: Res<GlobalTransforms>,
    mut solids: ResMut<Solids>,
) {
    let Some(limit) = collision.0.colliders() else {
        solids.limit = None;
        return;
    };
    let solids = &mut *solids;
    let since = globals.changes_since(&mut solids.moved);
    let settled = solids.limit == Some(limit)
        && solids.count == colliders.iter().len()
        && edited.is_empty()
        && match since {
            TransformsSince::Same => true,
            TransformsSince::Listed(moved) => solids.sync_moved(moved, &colliders, &globals),
            TransformsSince::All => false,
        };
    if !settled {
        solids.sync_all(limit, &colliders, &globals);
    }
}
