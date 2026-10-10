//! Engine edits systems queue on an entity: they need the whole world, so
//! they run where the schedule applies its commands, and a refused edit
//! fails as Bevy's failed commands do.

use bevy_ecs::prelude::*;

use crate::{Transform, edit_transform, reset_interpolation, set_parent};

/// The engine's edits of an entity, beside Bevy's own commands.
pub trait EngineEntityCommands {
    /// Gives the entity `parent` (none for a root) as [`set_parent`] does,
    /// which refuses cycles, missing entities and hierarchies past their
    /// limits.
    fn set_parent(&mut self, parent: Option<Entity>) -> &mut Self;

    /// Puts the entity at `transform` at once: frames show it there, not
    /// moving from where it was ([`edit_transform`], then
    /// [`reset_interpolation`] of its subtree).
    fn teleport(&mut self, transform: Transform) -> &mut Self;
}

impl EngineEntityCommands for EntityCommands<'_> {
    fn set_parent(&mut self, parent: Option<Entity>) -> &mut Self {
        self.queue(move |entity: EntityWorldMut| {
            let child = entity.id();
            set_parent(entity.into_world_mut(), child, parent)
        })
    }

    fn teleport(&mut self, transform: Transform) -> &mut Self {
        self.queue(move |entity: EntityWorldMut| {
            let id = entity.id();
            let world = entity.into_world_mut();
            edit_transform(world, id, |edited| *edited = transform)?;
            reset_interpolation(world, id);
            Ok::<(), crate::TransformEditError>(())
        })
    }
}
