//! Immediate edits of an existing local pose (decision 011).

use crate::{Transform, hierarchy::same_local_bits};
use bevy_ecs::prelude::*;
use std::fmt;

/// An edit of an entity that has no local pose to edit.
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

/// Immediately edits an existing local pose without advancing the simulation.
///
/// Writes through ordinary mutable component access, preserving the added tick
/// without invoking Insert observers. A bit-identical result leaves the change
/// tick untouched. Propagate transforms before querying derived globals; reset
/// interpolation for a teleport. Systems edit poses through ordinary
/// `Query<&mut Transform>`.
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
            .expect("editing a source copy cannot remove its world component") = next;
    }
    Ok(result)
}
