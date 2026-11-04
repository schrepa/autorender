use std::sync::Arc;

use bevy_ecs::prelude::Entity;
use engine_places::physics::{
    CollisionLayers, CollisionRay, PlacedPart, QueryFilter, RayHit, SensorFilter, ShapeCastOptions,
    ShapeHit,
};
use engine_places::solids::{Bounds, SolidField, WorldSolid};
use glam::Vec3;

use crate::geometry::{AffineShape, Narrow, intersects};
use crate::types::Fault;

fn accepts(filter: QueryFilter, solid: &WorldSolid) -> bool {
    filter.exclude != Some(solid.owner)
        && filter
            .layers
            .interacts(CollisionLayers::new(solid.memberships, solid.filter))
        && filter.sensors != SensorFilter::Only
}

fn candidates<'a>(
    sources: &'a [Arc<SolidField>],
    filter: QueryFilter,
    test: impl Fn(Bounds) -> bool + Copy + 'a,
) -> impl Iterator<Item = &'a WorldSolid> + 'a {
    sources
        .iter()
        .flat_map(move |source| source.solids.query(test))
        .filter(move |solid| accepts(filter, solid))
}

fn shaped(solid: &WorldSolid) -> Option<(&WorldSolid, AffineShape)> {
    Some((solid, AffineShape::new(solid.shape, solid.model).ok()?))
}

fn part(solid: &WorldSolid) -> PlacedPart {
    PlacedPart {
        layer: solid.layer,
        asset: solid.asset,
    }
}

pub(crate) fn ray(
    sources: &[Arc<SolidField>],
    ray: CollisionRay,
    filter: QueryFilter,
) -> Option<RayHit> {
    let mut nearest: Option<RayHit> = None;
    for (solid, affine) in
        candidates(sources, filter, move |bounds| bounds.ray(ray)).filter_map(shaped)
    {
        if let Some(hit) = hit(ray, solid, affine)
            && nearest.is_none_or(|old| hit.distance < old.distance)
        {
            nearest = Some(hit);
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
    for (solid, affine) in
        candidates(sources, filter, move |bounds| bounds.ray(ray)).filter_map(shaped)
    {
        if let Some(hit) = hit(ray, solid, affine) {
            output.push(hit);
        }
    }
}

pub(crate) fn overlap(
    sources: &[Arc<SolidField>],
    shape: &AffineShape,
    filter: QueryFilter,
    output: &mut Vec<Entity>,
) -> Result<(), Fault> {
    let reach = shape.bounds;
    for solid in candidates(sources, filter, move |bounds| bounds.overlaps(reach)) {
        if output.contains(&solid.owner) {
            continue;
        }
        let Some((_, affine)) = shaped(solid) else {
            continue;
        };
        if intersects(shape, &affine)? {
            output.push(solid.owner);
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
    for (solid, affine) in
        candidates(sources, filter, move |bounds| bounds.overlaps(swept)).filter_map(shaped)
    {
        if let Some((fraction, point, normal, started_overlapping)) =
            Narrow::cast(shape, &affine, displacement, options)?
            && nearest.is_none_or(|old| fraction < old.fraction)
        {
            nearest = Some(ShapeHit {
                entity: solid.owner,
                fraction,
                point,
                normal,
                started_overlapping,
                placed: Some(part(solid)),
            });
        }
    }
    Ok(nearest)
}

fn hit(ray: CollisionRay, solid: &WorldSolid, affine: AffineShape) -> Option<RayHit> {
    let (distance, normal, started_inside) = affine.ray(ray)?;
    Some(RayHit {
        entity: solid.owner,
        distance,
        point: ray.origin + ray.direction * distance,
        normal,
        started_inside,
        placed: Some(part(solid)),
    })
}
