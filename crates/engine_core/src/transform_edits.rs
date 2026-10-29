use crate::{Transform, hierarchy::same_local_bits};
use bevy_ecs::prelude::*;
use engine_settings::{Error, ErrorCode, Expected};

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
