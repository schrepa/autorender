use std::collections::BTreeMap;

use bevy_ecs::prelude::{Entity, Resource, World};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthoredId {
    pub scene: String,
    pub id: String,
}

#[derive(Resource, Clone)]
pub struct OperationScene {
    pub id: String,
    pub entities: BTreeMap<String, Entity>,
    pub streamed: bool,
    pub authored: fn(&World, Entity) -> Option<AuthoredId>,
}

impl OperationScene {
    pub fn entity(&self, world: &World, id: &str) -> Option<Entity> {
        self.entities.get(id).copied().or_else(|| {
            if !self.streamed {
                return None;
            }
            world.iter_entities().map(|entity| entity.id()).find(|&entity| {
                (self.authored)(world, entity)
                    .is_some_and(|authored| authored.id == id && authored.scene == self.id)
            })
        })
    }

    pub fn authored(&self, world: &World, entity: Entity) -> Option<AuthoredId> {
        (self.authored)(world, entity)
    }
}
