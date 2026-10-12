//! Terrain collision: a heightfield per level-0 terrain tile
//! resident around the camera and focus entities, of terrains that collide,
//! split into triangles exactly as renderers and terrain queries split them.
//! Rays, overlaps and shape casts test them after the colliders; hits report
//! the terrain's entity. Terrain belongs to the default collision layers
//! and is never a sensor.

use std::cell::Cell;

use bevy_ecs::prelude::Entity;
use engine_core::terrain_data::{TILE_QUADS, TILE_SAMPLES};
use std::sync::Arc;

use engine_core::{Terrains, Tile, TileKey};
use glam::Vec3;
use parry3d::{
    bounding_volume::Aabb,
    math::Pose,
    query::{
        Ray, RayCast, details,
        epa::EPA,
        gjk::{self, CsoPoint, GJKResult, VoronoiSimplex},
    },
    shape::{HeightField, HeightFieldCellStatus, Triangle},
    utils::Array2,
};

use crate::geometry::{AffineShape, Bounds, Support};
use crate::{
    CollisionError, CollisionLayers, CollisionRay, QueryFilter, RayHit, SensorFilter,
    ShapeCastOptions, ShapeHit,
};

/// One tile's heightfield: its terrain, key, the tile it was built from
/// (edits replace tiles under the same key), world center
/// (parry centers heightfields) and world bounds.
struct Field {
    entity: Entity,
    key: TileKey,
    tile: Arc<Tile>,
    center: Vec3,
    shape: HeightField,
    bounds: Bounds,
}

/// The heightfields of every colliding terrain's resident level-0 tiles.
#[derive(Default)]
pub(crate) struct TerrainFields {
    fields: Vec<Field>,
    revision: Option<u64>,
}

/// Whether `filter` accepts terrain `entity`.
fn accepts(filter: QueryFilter, entity: Entity) -> bool {
    filter.exclude != Some(entity)
        && filter.layers.interacts(CollisionLayers::default())
        && filter.sensors != SensorFilter::Only
}

impl TerrainFields {
    /// Heightfields in use.
    pub(crate) fn len(&self) -> usize {
        self.fields.len()
    }

    /// Follows `terrains`' collision tiles; returns whether they changed.
    pub(crate) fn sync(&mut self, terrains: Option<&Terrains>) -> bool {
        let Some(terrains) = terrains else {
            let changed = !self.fields.is_empty();
            self.fields.clear();
            self.revision = None;
            return changed;
        };
        let revision = terrains.revision();
        let wanted: Vec<_> = terrains.collision_tiles().collect();
        // The same tiles of the same residency: nothing to build. Tiles
        // already resident for rendering join without changing residency.
        let same = self.revision == Some(revision)
            && wanted.len() == self.fields.len()
            && wanted.iter().all(|(entity, _, tile)| {
                self.fields.iter().any(|field| {
                    field.entity == *entity
                        && field.key == tile.key()
                        && Arc::ptr_eq(&field.tile, tile)
                })
            });
        if same {
            return false;
        }
        self.revision = Some(revision);
        let mut kept = std::mem::take(&mut self.fields);
        for (entity, origin, tile) in wanted {
            let key = tile.key();
            if let Some(position) = kept.iter().position(|field| {
                field.entity == entity && field.key == key && Arc::ptr_eq(&field.tile, tile)
            }) {
                self.fields.push(kept.swap_remove(position));
                continue;
            }
            let samples = TILE_SAMPLES as usize;
            let spacing = tile.spacing();
            // Column-major: rows along z, columns along x.
            let mut heights = Vec::with_capacity(samples * samples);
            for x in 0..samples as i32 {
                for z in 0..samples as i32 {
                    heights.push(tile.height(x, z));
                }
            }
            let size = TILE_QUADS as f32 * spacing;
            let mut shape = HeightField::new(
                Array2::new(samples, samples, heights),
                Vec3::new(size, 1.0, size),
            );
            if tile.hole_bits().is_some() {
                for z in 0..TILE_QUADS {
                    for x in 0..TILE_QUADS {
                        if tile.is_hole(x, z) {
                            shape.set_cell_status(
                                z as usize,
                                x as usize,
                                HeightFieldCellStatus::CELL_REMOVED,
                            );
                        }
                    }
                }
            }
            let corner = Vec3::new(
                origin.x + f32::from(key.x) * size,
                origin.y,
                origin.z + f32::from(key.z) * size,
            );
            let [low, high] = tile.bounds();
            self.fields.push(Field {
                entity,
                key,
                tile: tile.clone(),
                center: corner + Vec3::new(size * 0.5, 0.0, size * 0.5),
                shape,
                bounds: Bounds {
                    min: Vec3::new(corner.x, origin.y + low, corner.z),
                    max: Vec3::new(corner.x + size, origin.y + high, corner.z + size),
                },
            });
        }
        true
    }

    /// The nearest terrain hit of `ray` (normalized) that `filter` accepts.
    pub(crate) fn ray(&self, ray: CollisionRay, filter: QueryFilter) -> Option<RayHit> {
        let mut nearest: Option<RayHit> = None;
        for field in &self.fields {
            if !accepts(filter, field.entity) || !field.bounds.ray(ray) {
                continue;
            }
            let local = Ray::new(ray.origin - field.center, ray.direction);
            let Some(hit) =
                field
                    .shape
                    .cast_local_ray_and_get_normal(&local, ray.max_distance, true)
            else {
                continue;
            };
            if !hit.time_of_impact.is_finite() {
                continue;
            }
            let candidate = RayHit {
                entity: field.entity,
                distance: hit.time_of_impact,
                point: ray.origin + ray.direction * hit.time_of_impact,
                normal: hit.normal.try_normalize().unwrap_or(Vec3::Y),
                started_inside: false,
                placed: None,
            };
            if nearest.is_none_or(|old| candidate.distance < old.distance) {
                nearest = Some(candidate);
            }
        }
        nearest
    }

    /// Every terrain hit of `ray`, one per terrain tile, into `output`.
    pub(crate) fn ray_all(&self, ray: CollisionRay, filter: QueryFilter, output: &mut Vec<RayHit>) {
        for field in &self.fields {
            if !accepts(filter, field.entity) || !field.bounds.ray(ray) {
                continue;
            }
            let local = Ray::new(ray.origin - field.center, ray.direction);
            if let Some(hit) = field
                .shape
                .cast_local_ray_and_get_normal(&local, ray.max_distance, true)
                .filter(|hit| hit.time_of_impact.is_finite())
            {
                output.push(RayHit {
                    entity: field.entity,
                    distance: hit.time_of_impact,
                    point: ray.origin + ray.direction * hit.time_of_impact,
                    normal: hit.normal.try_normalize().unwrap_or(Vec3::Y),
                    started_inside: false,
                    placed: None,
                });
            }
        }
    }

    /// Terrains whose triangles `shape` overlaps, into `output` once each.
    pub(crate) fn overlap(
        &self,
        shape: &AffineShape,
        filter: QueryFilter,
        output: &mut Vec<Entity>,
    ) -> Result<(), CollisionError> {
        for field in &self.fields {
            if !accepts(filter, field.entity)
                || output.contains(&field.entity)
                || !field.bounds.overlaps(shape.bounds)
            {
                continue;
            }
            let mut result = Ok(false);
            field.shape.map_elements_in_local_aabb(
                &local_aabb(shape.bounds, field.center),
                &mut |_, triangle| {
                    if matches!(result, Ok(false)) {
                        result = triangle_overlaps(shape, triangle, field.center);
                    }
                },
            );
            if result? {
                output.push(field.entity);
            }
        }
        Ok(())
    }

    /// The first terrain triangle `shape` meets moving by `displacement`.
    pub(crate) fn cast(
        &self,
        shape: &AffineShape,
        displacement: Vec3,
        filter: QueryFilter,
        options: ShapeCastOptions,
    ) -> Result<Option<ShapeHit>, CollisionError> {
        let swept = shape
            .bounds
            .swept(displacement)
            .expanded(options.target_distance);
        let mut nearest: Option<ShapeHit> = None;
        let mut epa = EPA::new();
        for field in &self.fields {
            if !accepts(filter, field.entity) || !field.bounds.overlaps(swept) {
                continue;
            }
            let mut failure = None;
            field.shape.map_elements_in_local_aabb(
                &local_aabb(swept, field.center),
                &mut |_, triangle| {
                    if failure.is_some() {
                        return;
                    }
                    match cast_triangle(
                        &mut epa,
                        shape,
                        triangle,
                        field.center,
                        displacement,
                        options,
                    ) {
                        Ok(Some((fraction, point, normal, started))) => {
                            if nearest.is_none_or(|old| fraction < old.fraction) {
                                nearest = Some(ShapeHit {
                                    entity: field.entity,
                                    fraction,
                                    point,
                                    normal,
                                    started_overlapping: started,
                                    placed: None,
                                });
                            }
                        }
                        Ok(None) => {}
                        Err(error) => failure = Some(error),
                    }
                },
            );
            if let Some(error) = failure {
                return Err(error);
            }
        }
        Ok(nearest)
    }
}

/// `bounds` in a field's local frame about `center`.
fn local_aabb(bounds: Bounds, center: Vec3) -> Aabb {
    Aabb::new(bounds.min - center, bounds.max - center)
}

/// `triangle` (local to `center`) relative to `shape`'s center.
fn relative(triangle: &Triangle, center: Vec3, shape: &AffineShape) -> Triangle {
    let offset = center - shape.center;
    Triangle::new(
        triangle.a + offset,
        triangle.b + offset,
        triangle.c + offset,
    )
}

/// GJK intersection of `shape` and `triangle`, as colliders test pairs.
fn triangle_overlaps(
    shape: &AffineShape,
    triangle: &Triangle,
    center: Vec3,
) -> Result<bool, CollisionError> {
    let calls = Cell::new(0);
    let triangle = relative(triangle, center, shape);
    let hit = details::intersection_test_support_map_support_map(
        &Pose::IDENTITY,
        &Support::new(shape, &calls),
        &triangle,
    );
    if calls.get() >= 200 {
        Err(CollisionError::NumericalFailure)
    } else {
        Ok(hit)
    }
}

/// When `shape` moving by `displacement` meets `triangle`: the fraction of
/// the motion, the contact point, the triangle's outward normal and whether
/// they started overlapping.
fn cast_triangle(
    epa: &mut EPA,
    shape: &AffineShape,
    triangle: &Triangle,
    center: Vec3,
    displacement: Vec3,
    options: ShapeCastOptions,
) -> Result<Option<(f32, Vec3, Vec3, bool)>, CollisionError> {
    let calls = Cell::new(0);
    let triangle = relative(triangle, center, shape);
    let support = Support::new(shape, &calls);
    let mut simplex = VoronoiSimplex::new();
    let toward = triangle.center().try_normalize().unwrap_or(Vec3::NEG_Y);
    simplex.reset(CsoPoint::from_shapes(
        &Pose::IDENTITY,
        &support,
        &triangle,
        toward,
    ));
    let proximity = gjk::closest_points(
        &Pose::IDENTITY,
        &support,
        &triangle,
        options.target_distance,
        true,
        &mut simplex,
    );
    if calls.get() >= 200 {
        return Err(CollisionError::NumericalFailure);
    }
    let contact = match proximity {
        GJKResult::Intersection => {
            epa.closest_points(&Pose::IDENTITY, &support, &triangle, &simplex)
        }
        GJKResult::ClosestPoints(p1, p2, normal) => Some((p1, p2, normal)),
        GJKResult::Proximity(_) | GJKResult::NoIntersection(_) => None,
    };
    if let Some((_, p2, _)) = contact {
        let normal = triangle.normal().unwrap_or(Vec3::Y);
        if !options.stop_at_penetration
            && displacement.dot(normal)
                >= -1.0e-6 * engine_core::math::max(displacement.length(), 1.0)
        {
            return Ok(None);
        }
        return Ok(Some((0.0, shape.center + p2, normal, true)));
    }
    if displacement == Vec3::ZERO {
        return Ok(None);
    }
    calls.set(0);
    let hit = details::cast_shapes_support_map_support_map(
        &Pose::IDENTITY,
        -displacement,
        &support,
        &triangle,
        parry3d::query::ShapeCastOptions {
            max_time_of_impact: 1.0,
            target_distance: options.target_distance,
            stop_at_penetration: true,
            compute_impact_geometry_on_penetration: false,
        },
    );
    if calls.get() >= 200 {
        return Err(CollisionError::NumericalFailure);
    }
    let Some(hit) = hit else {
        return Ok(None);
    };
    if hit.status != parry3d::query::ShapeCastStatus::Converged
        || !hit.time_of_impact.is_finite()
        || !(0.0..=1.0).contains(&hit.time_of_impact)
    {
        return Err(CollisionError::NumericalFailure);
    }
    Ok(Some((
        hit.time_of_impact,
        shape.center + hit.witness2,
        triangle.normal().unwrap_or(Vec3::Y),
        false,
    )))
}
