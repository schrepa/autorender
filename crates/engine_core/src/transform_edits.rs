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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_edits_preserve_lifetime_and_skip_bit_identical_values() {
        #[derive(Resource, Default)]
        struct Insertions(usize);

        let mut world = World::new();
        let entity = world.spawn(Transform::default()).id();
        world.init_resource::<Insertions>();
        world.add_observer(|_: On<Insert, Transform>, mut count: ResMut<Insertions>| {
            count.0 += 1;
        });
        world.flush();
        let added = world.entity(entity).get_ref::<Transform>().unwrap().added();
        world.increment_change_tick();
        assert_eq!(
            edit_transform(&mut world, entity, |pose| {
                pose.translation.x = f32::from_bits(0x7fc0_0042);
                7
            }),
            Ok(7)
        );
        let changed = world
            .entity(entity)
            .get_ref::<Transform>()
            .unwrap()
            .last_changed();
        world.increment_change_tick();
        edit_transform(&mut world, entity, |_| {}).unwrap();
        let pose = world.entity(entity).get_ref::<Transform>().unwrap();
        assert_eq!(pose.added(), added);
        assert_eq!(pose.last_changed(), changed);
        assert_ne!(changed, added);
        assert_eq!(pose.translation.x.to_bits(), 0x7fc0_0042);
        assert_eq!(world.resource::<Insertions>().0, 0);
        world.entity_mut(entity).remove::<Transform>();
        assert_eq!(
            edit_transform(&mut world, entity, |_| {}),
            Err(TransformEditError::MissingTransform(entity))
        );
        world.entity_mut(entity).despawn();
        assert_eq!(
            edit_transform(&mut world, entity, |_| {}),
            Err(TransformEditError::MissingEntity(entity))
        );
    }
}
