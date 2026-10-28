use bevy_ecs::prelude::{Entity, World};
use engine_core::GlobalTransforms;
use serde::Serialize;

use crate::{
    Collider, ColliderShape, ColliderStatus, CollisionError, CollisionWorld,
    validate_collider_transform,
};

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ColliderShapeView {
    Cuboid { half_extents: [f32; 3] },
    Sphere { radius: f32 },
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ColliderView {
    pub shape: ColliderShapeView,
    pub memberships: u32,
    pub filter: u32,
    pub sensor: bool,
    pub world_valid: bool,
    pub world_error: Option<String>,
    pub cache_status: String,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CollisionLimitsView {
    pub max_colliders: usize,
    pub max_candidate_pairs: usize,
    pub max_sensor_pairs: usize,
    pub max_query_candidates: usize,
    pub max_query_hits: usize,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CollisionWorldView {
    pub revision: String,
    pub frame: String,
    pub tick: String,
    pub colliders: usize,
    pub active_colliders: usize,
    pub invalid_colliders: usize,
    pub overflow_colliders: usize,
    pub bvh_nodes: usize,
    pub cache_bytes: usize,
    pub complete: bool,
    pub placed_colliders: usize,
    pub limits: CollisionLimitsView,
}

impl CollisionWorldView {
    pub fn of(value: &CollisionWorld) -> Self {
        let report = value.report();
        let config = value.config();
        Self {
            revision: report.revision.to_string(),
            frame: report.frame.to_string(),
            tick: report.tick.to_string(),
            colliders: report.colliders,
            active_colliders: report.active_colliders,
            invalid_colliders: report.invalid_colliders,
            overflow_colliders: report.overflow_colliders,
            bvh_nodes: report.bvh_nodes,
            cache_bytes: report.cache_bytes,
            complete: report.complete,
            placed_colliders: report.placed_colliders,
            limits: CollisionLimitsView {
                max_colliders: config.colliders(),
                max_candidate_pairs: config.candidate_pairs(),
                max_sensor_pairs: config.sensor_pairs(),
                max_query_candidates: config.query_candidates(),
                max_query_hits: config.query_hits(),
            },
        }
    }
}

pub fn collider_view(world: &World, entity: Entity) -> Option<ColliderView> {
    let collider = *world.get::<Collider>(entity)?;
    let valid = world
        .get_resource::<GlobalTransforms>()
        .and_then(|globals| globals.get(entity))
        .and_then(|global| global.matrix())
        .ok_or(CollisionError::InvalidTransform)
        .and_then(|matrix| validate_collider_transform(collider.shape(), matrix));
    let cache_status = world
        .get_resource::<CollisionWorld>()
        .map_or("unavailable", |cache| match cache.collider_status(entity) {
            Some(ColliderStatus::Active) => "active",
            Some(ColliderStatus::Invalid) => "invalid",
            Some(ColliderStatus::Overflow) => "overflow",
            None => "not_cached",
        });
    Some(ColliderView {
        shape: match collider.shape() {
            ColliderShape::Cuboid { half_extents } => ColliderShapeView::Cuboid {
                half_extents: half_extents.to_array(),
            },
            ColliderShape::Sphere { radius } => ColliderShapeView::Sphere { radius },
        },
        memberships: collider.layers().memberships,
        filter: collider.layers().filter,
        sensor: collider.is_sensor(),
        world_valid: valid.is_ok(),
        world_error: valid.err().map(|error| error.to_string()),
        cache_status: cache_status.to_owned(),
    })
}
