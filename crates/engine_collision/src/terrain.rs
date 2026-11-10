use std::cell::Cell;
use std::sync::Arc;

use bevy_ecs::prelude::Entity;
use engine_places_simulation::physics::{
    CollisionLayers, CollisionRay, QueryFilter, RayHit, SensorFilter, ShapeCastOptions, ShapeHit,
};
use engine_solids::{Bounds, FieldTile, SolidField};
use glam::{Vec2, Vec3};
use parry3d::{
    bounding_volume::Aabb,
    math::Pose,
    query::{
        Ray, RayCast, details,
        epa::EPA,
        gjk::{self, CsoPoint, GJKResult, VoronoiSimplex},
    },
    shape::Triangle,
};

use crate::geometry::{AffineShape, Support};
use crate::types::Fault;

fn tiles(sources: &[Arc<SolidField>]) -> impl Iterator<Item = &Arc<FieldTile>> {
    sources.iter().flat_map(|source| &source.fields)
}

fn floors(filter: QueryFilter) -> bool {
    filter.layers.interacts(CollisionLayers::default()) && filter.sensors != SensorFilter::Only
}

fn accepts(filter: QueryFilter, entity: Entity) -> bool {
    filter.exclude != Some(entity) && floors(filter)
}

fn covering(
    sources: &[Arc<SolidField>],
    at: Vec2,
) -> impl Iterator<Item = Option<&Arc<FieldTile>>> {
    sources
        .iter()
        .filter(move |source| !source.fields.iter().any(|field| field.covers(at)))
        .flat_map(move |source| {
            source
                .covers
                .iter()
                .filter(move |cover| cover.answers(at))
                .map(move |cover| cover.tile_at(at))
        })
}

pub(crate) fn awaits(sources: &[Arc<SolidField>], at: Vec2, filter: QueryFilter) -> bool {
    floors(filter) && covering(sources, at).any(|tile| tile.is_none())
}

pub(crate) fn below(
    sources: &[Arc<SolidField>],
    ray: CollisionRay,
    filter: QueryFilter,
) -> Option<RayHit> {
    if !floors(filter) {
        return None;
    }
    let at = Vec2::new(ray.origin.x, ray.origin.z);
    let mut nearest: Option<RayHit> = None;
    for field in covering(sources, at).flatten() {
        if !accepts(filter, field.owner) {
            continue;
        }
        let Some((height, normal)) = field.surface(at) else {
            continue;
        };
        let distance = ray.origin.y - height;
        if !(0.0..=ray.max_distance).contains(&distance) {
            continue;
        }
        let candidate = RayHit {
            entity: field.owner,
            distance,
            point: Vec3::new(at.x, height, at.y),
            normal,
            started_inside: false,
            placed: None,
        };
        if nearest.is_none_or(|old| candidate.distance < old.distance) {
            nearest = Some(candidate);
        }
    }
    nearest
}

pub(crate) fn ray(
    sources: &[Arc<SolidField>],
    ray: CollisionRay,
    filter: QueryFilter,
) -> Option<RayHit> {
    let mut nearest: Option<RayHit> = None;
    for field in tiles(sources) {
        if !accepts(filter, field.owner) || !field.bounds().ray(ray) {
            continue;
        }
        let local = Ray::new(ray.origin - field.center, ray.direction);
        let Some(hit) = field
            .shape()
            .cast_local_ray_and_get_normal(&local, ray.max_distance, true)
        else {
            continue;
        };
        if !hit.time_of_impact.is_finite() {
            continue;
        }
        let candidate = RayHit {
            entity: field.owner,
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

pub(crate) fn ray_all(
    sources: &[Arc<SolidField>],
    ray: CollisionRay,
    filter: QueryFilter,
    output: &mut Vec<RayHit>,
) {
    for field in tiles(sources) {
        if !accepts(filter, field.owner) || !field.bounds().ray(ray) {
            continue;
        }
        let local = Ray::new(ray.origin - field.center, ray.direction);
        if let Some(hit) = field
            .shape()
            .cast_local_ray_and_get_normal(&local, ray.max_distance, true)
            .filter(|hit| hit.time_of_impact.is_finite())
        {
            output.push(RayHit {
                entity: field.owner,
                distance: hit.time_of_impact,
                point: ray.origin + ray.direction * hit.time_of_impact,
                normal: hit.normal.try_normalize().unwrap_or(Vec3::Y),
                started_inside: false,
                placed: None,
            });
        }
    }
}

pub(crate) fn overlap(
    sources: &[Arc<SolidField>],
    shape: &AffineShape,
    filter: QueryFilter,
    output: &mut Vec<Entity>,
) -> Result<(), Fault> {
    for field in tiles(sources) {
        if !accepts(filter, field.owner)
            || output.contains(&field.owner)
            || !field.bounds().overlaps(shape.bounds)
        {
            continue;
        }
        let mut result = Ok(false);
        field.shape().map_elements_in_local_aabb(
            &local_aabb(shape.bounds, field.center),
            &mut |_, triangle| {
                if matches!(result, Ok(false)) {
                    result = triangle_overlaps(shape, triangle, field.center);
                }
            },
        );
        if result? {
            output.push(field.owner);
        }
    }
    Ok(())
}

pub(crate) fn cast(
    sources: &[Arc<SolidField>],
    shape: &AffineShape,
    displacement: Vec3,
    filter: QueryFilter,
    options: ShapeCastOptions,
) -> Result<Option<ShapeHit>, Fault> {
    let swept = shape
        .bounds
        .swept(displacement)
        .expanded(options.target_distance);
    let mut nearest: Option<ShapeHit> = None;
    let mut epa = EPA::new();
    for field in tiles(sources) {
        if !accepts(filter, field.owner) || !field.bounds().overlaps(swept) {
            continue;
        }
        let mut failure = None;
        field.shape().map_elements_in_local_aabb(
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
                                entity: field.owner,
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

fn local_aabb(bounds: Bounds, center: Vec3) -> Aabb {
    Aabb::new(bounds.min - center, bounds.max - center)
}

fn relative(triangle: &Triangle, center: Vec3, shape: &AffineShape) -> Triangle {
    let offset = center - shape.center;
    Triangle::new(
        triangle.a + offset,
        triangle.b + offset,
        triangle.c + offset,
    )
}

fn triangle_overlaps(
    shape: &AffineShape,
    triangle: &Triangle,
    center: Vec3,
) -> Result<bool, Fault> {
    let calls = Cell::new(0);
    let triangle = relative(triangle, center, shape);
    let hit = details::intersection_test_support_map_support_map(
        &Pose::IDENTITY,
        &Support::new(shape, &calls),
        &triangle,
    );
    if calls.get() >= 200 {
        Err(Fault::NumericalFailure)
    } else {
        Ok(hit)
    }
}

fn cast_triangle(
    epa: &mut EPA,
    shape: &AffineShape,
    triangle: &Triangle,
    center: Vec3,
    displacement: Vec3,
    options: ShapeCastOptions,
) -> Result<Option<(f32, Vec3, Vec3, bool)>, Fault> {
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
        return Err(Fault::NumericalFailure);
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
            && displacement.dot(normal) >= -1.0e-6 * engine_math::max(displacement.length(), 1.0)
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
        return Err(Fault::NumericalFailure);
    }
    let Some(hit) = hit else {
        return Ok(None);
    };
    if hit.status != parry3d::query::ShapeCastStatus::Converged
        || !hit.time_of_impact.is_finite()
        || !(0.0..=1.0).contains(&hit.time_of_impact)
    {
        return Err(Fault::NumericalFailure);
    }
    Ok(Some((
        hit.time_of_impact,
        shape.center + hit.witness2,
        triangle.normal().unwrap_or(Vec3::Y),
        false,
    )))
}
