use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::math::max;
use engine_core::{GlobalTransforms, Transform, TransformIndex, valid_global_matrix};
use engine_places::ground::{TILE_QUADS, TILE_SAMPLES, TileKey};
use engine_places::physics::{
    Collider, ColliderShape, Collision, CollisionLayers, MAX_COLLIDER_EXTENT,
    MAX_COLLISION_COORDINATE, MIN_COLLIDER_EXTENT, PlacedPart,
};
use engine_places::solids::{FieldSamples, FieldTile, SolidShape, SolidSource, WorldSolid};
use glam::{Mat3, Mat4, Vec3};
use parry3d::shape::{HeightField, HeightFieldCellStatus};
use parry3d::utils::Array2;

use crate::bounds::{Bounds, Bvh};

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
        let half = match shape {
            ColliderShape::Cuboid { .. } => {
                axes.x_axis.abs() + axes.y_axis.abs() + axes.z_axis.abs()
            }
            ColliderShape::Sphere { .. } => {
                let squared = axes.x_axis * axes.x_axis
                    + axes.y_axis * axes.y_axis
                    + axes.z_axis * axes.z_axis;
                Vec3::new(squared.x.sqrt(), squared.y.sqrt(), squared.z.sqrt())
            }
        };
        let margin = 16.0 * f32::EPSILON * max((center.abs() + half).max_element(), 1.0);
        let bounds = Bounds {
            min: center - half,
            max: center + half,
        }
        .expanded(margin);
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

pub(crate) struct Field {
    pub terrain: Entity,
    pub key: TileKey,
    samples: Arc<dyn FieldSamples>,
    pub center: Vec3,
    pub min: Vec3,
    pub max: Vec3,
    pub shape: Arc<HeightField>,
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
struct Input {
    entity: Entity,
    collider: Collider,
    revision: Option<u64>,
    hint: TransformIndex,
    active: bool,
}

#[derive(Clone, Copy)]
struct Entry {
    sensor: bool,
    found: ColliderSolid,
}

#[derive(Resource, Default)]
pub(crate) struct Solids {
    owners: Vec<Entity>,
    sources: u64,
    ground: Vec<Field>,
    placed: Vec<PlacedSolid>,
    placed_bounds: Vec<Bounds>,
    placed_bvh: Bvh,
    inputs: Vec<Input>,
    entries: Vec<Entry>,
    bounds: Vec<Bounds>,
    bvh: Bvh,
    count: usize,
    limit: Option<usize>,
    revision: u64,
}

impl Solids {
    pub fn collides(&self) -> bool {
        self.limit.is_some()
    }

    pub fn ground(&self) -> &[Field] {
        &self.ground
    }

    pub fn ground_revision(&self) -> u64 {
        self.sources
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn placed_in(&self, min: Vec3, max: Vec3) -> impl Iterator<Item = PlacedSolid> + '_ {
        let bounds = Bounds { min, max };
        self.placed_bvh
            .query(move |node| node.overlaps(bounds))
            .map(|index| self.placed[index])
            .filter(move |placed| placed.solid.bounds().overlaps(bounds))
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

    fn sync_ground<'a>(&mut self, wanted: impl Iterator<Item = &'a FieldTile>) {
        let mut kept = std::mem::take(&mut self.ground);
        for patch in wanted {
            let (entity, key, origin) = (patch.owner, patch.key, patch.origin);
            if let Some(position) = kept.iter().position(|field| {
                field.terrain == entity
                    && field.key == key
                    && (Arc::ptr_eq(&field.samples, &patch.samples)
                        || field.samples.same(&*patch.samples))
            }) {
                let mut field = kept.swap_remove(position);
                field.samples = patch.samples.clone();
                self.ground.push(field);
                continue;
            }
            let samples = TILE_SAMPLES as usize;
            let spacing = patch.spacing;
            let mut heights = Vec::with_capacity(samples * samples);
            for x in 0..samples as i32 {
                for z in 0..samples as i32 {
                    heights.push(patch.samples.height(x, z));
                }
            }
            let size = TILE_QUADS as f32 * spacing;
            let mut shape = HeightField::new(
                Array2::new(samples, samples, heights),
                Vec3::new(size, 1.0, size),
            );
            for z in 0..TILE_QUADS {
                for x in 0..TILE_QUADS {
                    if patch.samples.hole(x, z) {
                        shape.set_cell_status(
                            z as usize,
                            x as usize,
                            HeightFieldCellStatus::CELL_REMOVED,
                        );
                    }
                }
            }
            let corner = Vec3::new(
                origin.x + f32::from(key.x) * size,
                origin.y,
                origin.z + f32::from(key.z) * size,
            );
            let [low, high] = patch.bounds;
            self.ground.push(Field {
                terrain: entity,
                key,
                samples: patch.samples.clone(),
                center: corner + Vec3::new(size * 0.5, 0.0, size * 0.5),
                min: Vec3::new(corner.x, origin.y + low, corner.z),
                max: Vec3::new(corner.x + size, origin.y + high, corner.z + size),
                shape: Arc::new(shape),
            });
        }
    }

    fn sync_placed<'a>(&mut self, solids: impl Iterator<Item = &'a WorldSolid>) {
        self.placed.clear();
        for solid in solids {
            let shape = match solid.shape {
                SolidShape::Cuboid { half_extents } => ColliderShape::Cuboid { half_extents },
                SolidShape::Sphere { radius } => ColliderShape::Sphere { radius },
            };
            let Some(found) = Solid::new(shape, solid.model) else {
                continue;
            };
            self.placed.push(PlacedSolid {
                terrain: solid.owner,
                part: PlacedPart {
                    layer: solid.layer,
                    asset: solid.asset,
                },
                solid: found,
            });
        }
        self.placed_bounds.clear();
        self.placed_bounds
            .extend(self.placed.iter().map(|placed| placed.solid.bounds()));
        self.placed_bvh.rebuild(&self.placed_bounds);
    }
}

fn read_pose(
    globals: &GlobalTransforms,
    entity: Entity,
    has_local: bool,
    hint: &mut TransformIndex,
) -> (Option<Mat4>, Option<u64>) {
    if !has_local {
        return (None, None);
    }
    globals
        .get_indexed_with_revision(entity, hint)
        .map_or((None, None), |(pose, revision)| {
            (pose.matrix(), Some(revision))
        })
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
    solids
        .owners
        .extend(sources.iter().map(|(entity, _)| entity));
    solids.sources = solids.sources.wrapping_add(1);
    solids.sync_ground(
        sources
            .iter()
            .flat_map(|(_, source)| source.into_inner().field.fields()),
    );
    solids.sync_placed(
        sources
            .iter()
            .flat_map(|(_, source)| source.into_inner().field.solids()),
    );
}

pub(crate) fn sync_colliders(
    collision: Res<Collision>,
    colliders: Query<(Entity, &Collider, Option<&Transform>)>,
    globals: Res<GlobalTransforms>,
    mut solids: ResMut<Solids>,
) {
    let Some(limit) = collision.0.colliders() else {
        solids.limit = None;
        return;
    };
    let count = colliders.iter().len();
    let admitted = count.min(limit);
    let unchanged = solids.revision != 0
        && solids.limit == Some(limit)
        && count == solids.count
        && solids.inputs.len() == admitted
        && colliders.iter().take(limit).zip(&mut solids.inputs).all(
            |((entity, collider, local), old)| {
                old.entity == entity
                    && old.collider == *collider
                    && old.revision != Some(u64::MAX)
                    && read_pose(&globals, entity, local.is_some(), &mut old.hint).1
                        == old.revision
            },
        );
    if unchanged {
        return;
    }
    let solids = &mut *solids;
    solids.revision = solids.revision.saturating_add(1);
    solids.limit = Some(limit);
    solids.count = count;
    solids.entries.clear();
    solids.bounds.clear();
    let mut statuses_changed = solids.inputs.len() != admitted;
    solids.inputs.truncate(admitted);
    for (index, (entity, collider, local)) in colliders.iter().take(limit).enumerate() {
        let previous = solids
            .inputs
            .get(index)
            .copied()
            .filter(|old| old.entity == entity);
        let mut hint = previous.map_or_else(TransformIndex::default, |old| old.hint);
        let (matrix, revision) = read_pose(&globals, entity, local.is_some(), &mut hint);
        let solid = matrix.and_then(|matrix| Solid::new(collider.shape(), matrix));
        if let Some(solid) = solid {
            solids.entries.push(Entry {
                sensor: collider.is_sensor(),
                found: ColliderSolid {
                    entity,
                    layers: collider.layers(),
                    solid,
                },
            });
            solids.bounds.push(solid.bounds());
        }
        let input = Input {
            entity,
            collider: *collider,
            revision,
            hint,
            active: solid.is_some(),
        };
        if let Some(old) = solids.inputs.get_mut(index) {
            statuses_changed |= old.entity != entity || old.active != input.active;
            *old = input;
        } else {
            statuses_changed = true;
            solids.inputs.push(input);
        }
    }
    solids.bvh.update(&solids.bounds, !statuses_changed);
}
