use bevy_ecs::prelude::{Entity, World};
use engine_places::{SolidShape, WorldSolid, WorldSolids};
use glam::Vec3;

use crate::bvh::Bvh;
use crate::geometry::{AffineShape, Bounds, Narrow, intersects};
use crate::{
    ColliderShape, CollisionLayers, CollisionRay, Fault, PlacedPart, QueryFilter, RayHit,
    SensorFilter, ShapeCastOptions, ShapeHit,
};

#[derive(Clone, Copy)]
struct Object {
    entity: Entity,
    part: PlacedPart,
    layers: CollisionLayers,
    affine: AffineShape,
}

#[derive(Default)]
pub(crate) struct PlacedShapes {
    objects: Vec<Object>,
    bounds: Vec<Bounds>,
    bvh: Bvh,
    revision: Option<u64>,
}

fn accepts(filter: QueryFilter, object: &Object) -> bool {
    filter.exclude != Some(object.entity)
        && filter.layers.interacts(object.layers)
        && filter.sensors != SensorFilter::Only
}

impl PlacedShapes {
    pub(crate) fn len(&self) -> usize {
        self.objects.len()
    }

    pub(crate) fn objects_in(
        &self,
        bounds: Bounds,
    ) -> impl Iterator<Item = (Entity, PlacedPart, &AffineShape)> + '_ {
        self.bvh
            .query(move |node| node.overlaps(bounds))
            .map(|index| &self.objects[index])
            .filter(move |object| object.affine.bounds.overlaps(bounds))
            .map(|object| (object.entity, object.part, &object.affine))
    }

    pub(crate) fn sync(&mut self, world: &World) -> bool {
        let solids = world.get_resource::<WorldSolids>();
        let revision = solids.map(|solids| solids.revision(world));
        if revision == self.revision {
            return false;
        }
        self.revision = revision;
        let had = !self.objects.is_empty();
        self.objects.clear();
        if let Some(solids) = solids {
            self.collect(solids.solids(world));
        }
        self.bounds.clear();
        self.bounds
            .extend(self.objects.iter().map(|object| object.affine.bounds));
        self.bvh.rebuild(&self.bounds);
        had || !self.objects.is_empty()
    }

    fn collect(&mut self, solids: Vec<WorldSolid>) {
        for solid in solids {
            let shape = match solid.shape {
                SolidShape::Cuboid { half_extents } => ColliderShape::Cuboid { half_extents },
                SolidShape::Sphere { radius } => ColliderShape::Sphere { radius },
            };
            let Ok(affine) = AffineShape::new(shape, solid.model) else {
                continue;
            };
            self.objects.push(Object {
                entity: solid.owner,
                part: PlacedPart {
                    layer: solid.layer,
                    asset: solid.asset,
                },
                layers: CollisionLayers::new(solid.memberships, solid.filter),
                affine,
            });
        }
    }

    pub(crate) fn ray(&self, ray: CollisionRay, filter: QueryFilter) -> Option<RayHit> {
        let mut nearest: Option<RayHit> = None;
        for index in self.bvh.query(|bounds| bounds.ray(ray)) {
            let object = &self.objects[index];
            if !accepts(filter, object) {
                continue;
            }
            if let Some(hit) = hit(ray, object)
                && nearest.is_none_or(|old| hit.distance < old.distance)
            {
                nearest = Some(hit);
            }
        }
        nearest
    }

    pub(crate) fn ray_all(&self, ray: CollisionRay, filter: QueryFilter, output: &mut Vec<RayHit>) {
        for index in self.bvh.query(|bounds| bounds.ray(ray)) {
            let object = &self.objects[index];
            if accepts(filter, object)
                && let Some(hit) = hit(ray, object)
            {
                output.push(hit);
            }
        }
    }

    pub(crate) fn overlap(
        &self,
        shape: &AffineShape,
        filter: QueryFilter,
        output: &mut Vec<Entity>,
    ) -> Result<(), Fault> {
        for index in self.bvh.query(|bounds| bounds.overlaps(shape.bounds)) {
            let object = &self.objects[index];
            if !accepts(filter, object) || output.contains(&object.entity) {
                continue;
            }
            if intersects(shape, &object.affine)? {
                output.push(object.entity);
            }
        }
        Ok(())
    }

    pub(crate) fn cast(
        &self,
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
        for index in self.bvh.query(|bounds| bounds.overlaps(swept)) {
            let object = &self.objects[index];
            if !accepts(filter, object) {
                continue;
            }
            if let Some((fraction, point, normal, started_overlapping)) =
                Narrow::cast(shape, &object.affine, displacement, options)?
                && nearest.is_none_or(|old| fraction < old.fraction)
            {
                nearest = Some(ShapeHit {
                    entity: object.entity,
                    fraction,
                    point,
                    normal,
                    started_overlapping,
                    placed: Some(object.part),
                });
            }
        }
        Ok(nearest)
    }
}

fn hit(ray: CollisionRay, object: &Object) -> Option<RayHit> {
    let (distance, normal, started_inside) = object.affine.ray(ray)?;
    Some(RayHit {
        entity: object.entity,
        distance,
        point: ray.origin + ray.direction * distance,
        normal,
        started_inside,
        placed: Some(object.part),
    })
}
