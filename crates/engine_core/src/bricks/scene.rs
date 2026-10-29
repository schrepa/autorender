use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::{Entity, Resource, World};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthoredId {
    pub scene: String,
    pub id: String,
    pub name: String,
}

#[derive(Resource, Clone)]
pub struct OperationScene {
    pub id: String,
    pub entities: Arc<BTreeMap<String, Entity>>,
    pub streamed: bool,
    pub authored: fn(&World, Entity) -> Option<AuthoredId>,
}

impl OperationScene {
    pub fn entity(&self, world: &World, id: &str) -> Option<Entity> {
        self.entities.get(id).copied().or_else(|| {
            if !self.streamed {
                return None;
            }
            world
                .iter_entities()
                .map(|entity| entity.id())
                .find(|&entity| {
                    (self.authored)(world, entity)
                        .is_some_and(|authored| authored.id == id && authored.scene == self.id)
                })
        })
    }

    pub fn authored(&self, world: &World, entity: Entity) -> Option<AuthoredId> {
        (self.authored)(world, entity)
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn holds(&self, authored: &AuthoredId, entity: Entity) -> bool {
        authored.scene == self.id
            && (self.streamed || self.entities.get(&authored.id) == Some(&entity))
    }
}

pub fn authored_entity(world: &World, id: &str) -> Option<Entity> {
    world.get_resource::<OperationScene>()?.entity(world, id)
}

pub fn authored_id(world: &World, entity: Entity) -> Option<String> {
    world
        .get_resource::<OperationScene>()?
        .authored(world, entity)
        .map(|authored| authored.id)
}
