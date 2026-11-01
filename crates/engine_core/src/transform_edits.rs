use crate::{Transform, hierarchy::same_local_bits};
use bevy_ecs::prelude::*;
use engine_registry::OperationScene;
use engine_settings::{Error, ErrorCode, Expected};
use serde_json::{Value, json};

pub(crate) const EDIT_TRANSFORM: &str = "edit_transform";

pub fn edit_transform<R>(
    world: &mut World,
    entity: Entity,
    edit: impl FnOnce(&mut Transform) -> R,
) -> Result<R, Error> {
    let source = world.get::<Transform>(entity).copied().ok_or_else(|| {
        if world.entities().contains(entity) {
            Error::new(ErrorCode::MissingComponent, "$.entity.transform")
                .expected(Expected::Present)
                .got(entity.to_bits())
        } else {
            Error::new(ErrorCode::UnknownEntity, "$.entity")
                .expected(Expected::Live)
                .got(entity.to_bits())
        }
    })?;
    let mut next = source;
    let result = edit(&mut next);
    if !same_local_bits(&source, &next) {
        *world
            .get_mut::<Transform>(entity)
            .expect("world_component_kept") = next;
    }
    Ok(result)
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct EditTransform {
    entity: String,
    translation: [f32; 3],
}

pub(crate) fn run_edit_transform(world: &mut World, request: Value) -> Result<Value, Error> {
    let request: EditTransform = engine_settings::parse_value(request)?;
    let entity = world
        .get_resource::<OperationScene>()
        .and_then(|scene| scene.entity(world, &request.entity))
        .ok_or_else(|| Error::new(ErrorCode::UnknownEntity, "$.entity").got(&request.entity))?;
    let translation = glam::Vec3::from_array(request.translation);
    edit_transform(world, entity, |transform| {
        transform.translation = translation
    })?;
    Ok(json!({}))
}
