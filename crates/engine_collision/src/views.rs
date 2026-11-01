use bevy_ecs::prelude::*;
use engine_core::{Error, ErrorCode, GlobalTransforms, Operation, SceneIds};
use engine_places::physics::{Collider, ColliderShape, Collision};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::queries::{reply, resolve_entity, schema_of};
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

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ColliderQuery {
    pub entity: String,
}

impl Operation for ColliderQuery {
    const NAME: &'static str = "collider";
    type Params = (
        SceneIds<'static, 'static>,
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
            .ok_or_else(|| {
                Error::new(ErrorCode::MissingComponent, "$.entity").got(&request.entity)
            })?;
        Ok(reply(view))
    }

    fn schema() -> Value {
        schema_of::<Self>()
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
