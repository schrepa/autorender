//! Placed objects' colliders: what terrains' placement layers
//! stand within their collision radius, each object its
//! asset's primitive at its pose, its base on the finest resident heights
//! as renderers set it. Rays, overlaps and shape casts test them after the
//! colliders; hits report the terrain's entity and the object's placement
//! layer and asset. Objects standing far away never collide.
//! Bridges' decks collide as a box a deck segment, reporting
//! layer [`BRIDGE_LAYER`] and their bridge's index as the asset.

use engine_core::collections::HashMap;
use std::sync::Arc;

use bevy_ecs::prelude::Entity;
use engine_core::terrain_data::TILE_QUADS;
use engine_core::{DECK_DEPTH, DECK_MARGIN, DECK_RISE, PlacedShape, Terrains, Tile};
use glam::{Mat4, Quat, Vec2, Vec3};

use crate::bvh::Bvh;
use crate::geometry::{AffineShape, Bounds, Narrow, intersects};
use crate::{
    ColliderShape, CollisionError, CollisionLayers, CollisionRay, PlacedPart, QueryFilter, RayHit,
    SensorFilter, ShapeCastOptions, ShapeHit,
};

/// The placement layer bridges' decks report hits in.
pub const BRIDGE_LAYER: u16 = u16::MAX;

#[derive(Clone, Copy)]
struct Object {
    entity: Entity,
    part: PlacedPart,
    layers: CollisionLayers,
    affine: AffineShape,
}

/// The colliders of every colliding terrain's placed objects within its
/// collision radius.
#[derive(Default)]
pub(crate) struct PlacedShapes {
    objects: Vec<Object>,
    bounds: Vec<Bounds>,
    bvh: Bvh,
    revision: Option<u64>,
}

/// A colliding terrain's level-0 tiles around the camera and focus
/// entities, by column and row, with its origin and their size.
struct Ground<'a> {
    origin: Vec3,
    size: f32,
    tiles: HashMap<(u16, u16), &'a Arc<Tile>>,
}

/// Whether `filter` accepts `object`: never as a sensor.
fn accepts(filter: QueryFilter, object: &Object) -> bool {
    filter.exclude != Some(object.entity)
        && filter.layers.interacts(object.layers)
        && filter.sensors != SensorFilter::Only
}

impl PlacedShapes {
    /// Objects colliding.
    pub(crate) fn len(&self) -> usize {
        self.objects.len()
    }

    /// The objects whose bounds meet `bounds`: each one's terrain,
    /// placement and shape.
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

    /// Follows `terrains`' resident tiles; returns whether the objects
    /// changed.
    pub(crate) fn sync(&mut self, terrains: Option<&Terrains>) -> bool {
        let revision = terrains.map(Terrains::revision);
        if revision == self.revision {
            return false;
        }
        self.revision = revision;
        let had = !self.objects.is_empty();
        self.objects.clear();
        if let Some(terrains) = terrains {
            self.collect(terrains);
        }
        self.bounds.clear();
        self.bounds
            .extend(self.objects.iter().map(|object| object.affine.bounds));
        self.bvh.rebuild(&self.bounds);
        had || !self.objects.is_empty()
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn collect(&mut self, terrains: &Terrains) {
        // Bridges' decks: a box a segment, as the deck's slab and road stand.
        for (entity, origin, bridges) in terrains.colliding_bridges() {
            for (index, bridge) in bridges.iter().enumerate() {
                for pair in bridge.deck.windows(2) {
                    let (a, b) = (origin + pair[0], origin + pair[1]);
                    let along = b - a;
                    let length = along.length();
                    if length < 1.0e-3 {
                        continue;
                    }
                    let center = 0.5 * (a + b) + Vec3::Y * (0.5 * (DECK_RISE - DECK_DEPTH));
                    let model = Mat4::from_rotation_translation(
                        Quat::from_rotation_arc(Vec3::X, along / length),
                        center,
                    );
                    let shape = ColliderShape::Cuboid {
                        half_extents: Vec3::new(
                            0.5 * length,
                            0.5 * (DECK_RISE + DECK_DEPTH),
                            0.5 * bridge.width + DECK_MARGIN,
                        ),
                    };
                    let Ok(affine) = AffineShape::new(shape, model) else {
                        continue;
                    };
                    self.objects.push(Object {
                        entity,
                        part: PlacedPart {
                            layer: BRIDGE_LAYER,
                            asset: index.min(usize::from(u16::MAX)) as u16,
                        },
                        layers: CollisionLayers::default(),
                        affine,
                    });
                }
            }
        }
        let mut grounds: HashMap<Entity, Ground<'_>> = HashMap::default();
        for (entity, origin, tile) in terrains.collision_tiles() {
            let key = tile.key();
            grounds
                .entry(entity)
                .or_insert_with(|| Ground {
                    origin,
                    size: TILE_QUADS as f32 * tile.spacing(),
                    tiles: HashMap::default(),
                })
                .tiles
                .insert((key.x, key.z), tile);
        }
        for (
            entity,
            Ground {
                origin,
                size,
                tiles,
            },
        ) in grounds
        {
            let Some(terrain) = terrains.terrain(entity) else {
                continue;
            };
            let assets = &terrain.surface.assets;
            if assets.iter().all(|asset| asset.collider.is_none()) {
                continue;
            }
            // Only tiles reaching the collision tiles hold objects that
            // collide.
            let (low, high) = tiles.keys().fold(
                (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)),
                |(low, high), &(x, z)| {
                    let corner = Vec2::new(f32::from(x), f32::from(z)) * size;
                    (low.min(corner), high.max(corner + size))
                },
            );
            for tile in terrains.tiles(entity) {
                let key = tile.key();
                let reach = TILE_QUADS as f32 * tile.spacing();
                let corner = Vec2::new(f32::from(key.x), f32::from(key.z)) * reach;
                if tile.placed().is_empty()
                    || (corner + reach).cmplt(low).any()
                    || corner.cmpgt(high).any()
                {
                    continue;
                }
                for placed in tile.placed() {
                    if placed.near_level.is_some() {
                        continue;
                    }
                    let Some(collider) = assets
                        .get(usize::from(placed.asset))
                        .and_then(|asset| asset.collider)
                    else {
                        continue;
                    };
                    let [x, _, z] = placed.position;
                    let (column, row) = ((x / size).floor(), (z / size).floor());
                    if !(0.0..65536.0).contains(&column) || !(0.0..65536.0).contains(&row) {
                        continue;
                    }
                    let Some(ground) = tiles.get(&(column as u16, row as u16)) else {
                        continue;
                    };
                    let local = Vec2::new(x - column * size, z - row * size);
                    let Some(sample) = ground.sample(local).filter(|sample| !sample.hole) else {
                        continue;
                    };
                    let model = placed.model(origin, sample.height - placed.sink)
                        * Mat4::from_translation(Vec3::from_array(collider.center));
                    let shape = match collider.shape {
                        PlacedShape::Cuboid { half_extents } => ColliderShape::Cuboid {
                            half_extents: Vec3::from_array(half_extents),
                        },
                        PlacedShape::Sphere { radius } => ColliderShape::Sphere { radius },
                    };
                    let Ok(affine) = AffineShape::new(shape, model) else {
                        continue;
                    };
                    self.objects.push(Object {
                        entity,
                        part: PlacedPart {
                            layer: placed.layer,
                            asset: placed.asset,
                        },
                        layers: CollisionLayers::new(collider.memberships, collider.filter),
                        affine,
                    });
                }
            }
        }
    }

    /// The nearest object hit of `ray` (normalized) that `filter` accepts.
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

    /// Every object hit of `ray`, into `output`.
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

    /// The terrains whose objects `shape` overlaps, into `output` once each.
    pub(crate) fn overlap(
        &self,
        shape: &AffineShape,
        filter: QueryFilter,
        output: &mut Vec<Entity>,
    ) -> Result<(), CollisionError> {
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

    /// The first object `shape` meets moving by `displacement`.
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

/// `ray`'s hit of `object`, if any.
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
