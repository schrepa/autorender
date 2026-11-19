use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_settings::{Error, ErrorCode, Expected};

#[derive(Component, Debug)]
pub struct SceneEntity {
    pub scene_id: String,
    pub authored_id: String,
    pub name: String,
    pub tags: Vec<String>,
    pub authored_parent: Option<String>,
    pub prefab_id: Option<String>,
    pub prefab_instance: Option<String>,
    pub prefab_local_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthoredId {
    pub scene: String,
    pub id: String,
    pub name: String,
}

impl SceneEntity {
    pub fn authored(&self) -> AuthoredId {
        AuthoredId {
            scene: self.scene_id.clone(),
            id: self.authored_id.clone(),
            name: self.name.clone(),
        }
    }
}

#[derive(Resource, Clone)]
pub struct OperationScene {
    pub id: String,
    pub entities: Arc<BTreeMap<String, Entity>>,
    pub installed: BTreeMap<String, Entity>,
    removed: BTreeSet<String>,
}

impl OperationScene {
    pub fn new(id: String, entities: Arc<BTreeMap<String, Entity>>) -> Self {
        Self {
            id,
            entities,
            installed: BTreeMap::new(),
            removed: BTreeSet::new(),
        }
    }

    pub fn entity(&self, id: &str) -> Option<Entity> {
        self.entities
            .get(id)
            .or_else(|| self.installed.get(id))
            .copied()
    }

    pub fn named(
        &self,
        id: &str,
        path: &str,
        live: impl Fn(Entity) -> bool,
    ) -> Result<Entity, Error> {
        match self.entity(id) {
            Some(entity) if live(entity) => Ok(entity),
            Some(_) => Err(entity_removed(id, path)),
            None if self.removed.contains(id) => Err(entity_removed(id, path)),
            None => Err(Error::new(ErrorCode::UnknownEntity, path).got(id)),
        }
    }

    pub fn named_in(&self, world: &World, id: &str, path: &str) -> Result<Entity, Error> {
        self.named(id, path, |entity| world.get_entity(entity).is_ok())
    }

    pub fn authored(&self, world: &World, entity: Entity) -> Option<AuthoredId> {
        world.get::<SceneEntity>(entity).map(SceneEntity::authored)
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len() + self.installed.len()
    }

    pub fn holds(&self, authored: &AuthoredId, entity: Entity) -> bool {
        authored.scene == self.id && self.entity(&authored.id) == Some(entity)
    }

    pub fn install(&mut self, authored: &SceneEntity, entity: Entity) {
        if authored.scene_id == self.id && self.entities.get(&authored.authored_id) != Some(&entity)
        {
            self.removed.remove(&authored.authored_id);
            self.installed.insert(authored.authored_id.clone(), entity);
        }
    }

    pub fn uninstall(&mut self, authored: &SceneEntity, entity: Entity) {
        if authored.scene_id == self.id
            && self.installed.get(&authored.authored_id) == Some(&entity)
        {
            self.installed.remove(&authored.authored_id);
            self.removed.insert(authored.authored_id.clone());
        }
    }

    pub fn forget(&mut self, id: &str) {
        self.removed.remove(id);
    }
}

pub fn entity_removed(id: &str, path: &str) -> Error {
    Error::new(ErrorCode::EntityRemoved, path)
        .expected(Expected::Live)
        .got(id)
}
