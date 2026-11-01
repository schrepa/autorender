use bevy_ecs::prelude::*;
use engine_core::{Error, ErrorCode, GlobalTransforms, Operation, SceneIds};
use engine_places::physics::{Collider, ColliderShape, Collision};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::queries::{reply, resolve_entity};
use crate::types::{ColliderStatus, Fault, check_collider_transform};
use crate::world::{CollisionWorld, world_of};

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ColliderShapeView {
    Cuboid { half_extents: [f32; 3] },
    Sphere { radius: f32 },
}

#[derive(Serialize)]
pub(crate) struct ColliderView {
    pub shape: ColliderShapeView,
    pub memberships: u32,
    pub filter: u32,
    pub sensor: bool,
    pub world_valid: bool,
    pub world_error: Option<Error>,
    pub cache_status: String,
}

#[derive(Serialize)]
pub(crate) struct CollisionLimitsView {
    pub max_colliders: usize,
    pub max_candidate_pairs: usize,
    pub max_sensor_pairs: usize,
    pub max_query_candidates: usize,
    pub max_query_hits: usize,
}

#[derive(Serialize)]
pub(crate) struct CollisionWorldView {
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
    pub(crate) fn of(value: &CollisionWorld) -> Self {
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

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ColliderQuery {
    pub entity: String,
}

impl Operation for ColliderQuery {
    const NAME: &'static str = "collider";
    type Params = (
        Res<'static, SceneIds>,
        Query<'static, 'static, Entity>,
        Query<'static, 'static, &'static Collider>,
        Res<'static, GlobalTransforms>,
        Res<'static, Collision>,
    );
    type Request = Self;

    fn run(
        (ids, entities, colliders, globals, collision): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        let entity = resolve_entity(&ids, &entities, &request.entity, false, "$.entity")?;
        let view = collider_view(&colliders, &globals, world_of(&collision), entity)
            .ok_or_else(|| Error::new(ErrorCode::MissingComponent, "$.entity").got(&request.entity))?;
        Ok(reply(view))
    }
}

fn collider_view(
    colliders: &Query<&Collider>,
    globals: &GlobalTransforms,
    cache: Option<&CollisionWorld>,
    entity: Entity,
) -> Option<ColliderView> {
    let collider = *colliders.get(entity).ok()?;
    let valid = globals
        .get(entity)
        .and_then(|global| global.matrix())
        .ok_or(Fault::InvalidTransform)
        .and_then(|matrix| check_collider_transform(collider.shape(), matrix));
    let cache_status = cache.map_or("unavailable", |cache| match cache.collider_status(entity) {
        Some(ColliderStatus::Active) => "active",
        Some(ColliderStatus::Invalid) => "invalid",
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
        world_error: valid.err().map(Fault::error),
        cache_status: cache_status.to_owned(),
    })
}
