use bevy_ecs::prelude::*;

use crate::{Transform, edit_transform, reset_interpolation, set_parent};

pub trait EngineEntityCommands {
    fn set_parent(&mut self, parent: Option<Entity>) -> &mut Self;

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
