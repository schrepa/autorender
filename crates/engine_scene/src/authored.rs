use engine_core::bevy_ecs::prelude::{Entity, World};

use crate::{SceneEntity, SceneInstance};

impl SceneInstance {
    pub fn authored_entity(&self, world: &World, id: &str) -> Option<Entity> {
        self.entity(id).or_else(|| {
            if !self.streamed() {
                return None;
            }
            world
                .try_query::<(Entity, &SceneEntity)>()?
                .iter(world)
                .find(|(_, metadata)| metadata.authored_id == id && metadata.scene_id == self.id())
                .map(|(entity, _)| entity)
        })
    }
}
