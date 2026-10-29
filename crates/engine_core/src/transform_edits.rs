use crate::{Transform, hierarchy::same_local_bits};
use bevy_ecs::prelude::*;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformEditError {
    MissingEntity(Entity),
    MissingTransform(Entity),
}

impl fmt::Display for TransformEditError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingEntity(entity) => write!(formatter, "entity {entity:?} does not exist"),
            Self::MissingTransform(entity) => {
                write!(formatter, "entity {entity:?} has no local Transform")
            }
        }
    }
}

impl std::error::Error for TransformEditError {}

pub fn edit_transform<R>(
    world: &mut World,
    entity: Entity,
    edit: impl FnOnce(&mut Transform) -> R,
) -> Result<R, TransformEditError> {
    let source = world.get::<Transform>(entity).copied().ok_or_else(|| {
        if world.entities().contains(entity) {
            TransformEditError::MissingTransform(entity)
        } else {
            TransformEditError::MissingEntity(entity)
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
