use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_place::{EntityIds, Error};
use serde_json::Value;

#[derive(Clone, Copy)]
pub struct SceneRead(pub fn(&mut World, Entity, &Value) -> Result<(), Error>);

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
}

impl OperationScene {
    pub fn entity(&self, id: &str) -> Option<Entity> {
        self.entities
            .get(id)
            .or_else(|| self.installed.get(id))
            .copied()
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
            self.installed.insert(authored.authored_id.clone(), entity);
        }
    }

    pub fn uninstall(&mut self, authored: &SceneEntity, entity: Entity) {
        if authored.scene_id == self.id
            && self.installed.get(&authored.authored_id) == Some(&entity)
        {
            self.installed.remove(&authored.authored_id);
        }
    }
}

pub fn authored_entity(world: &World, id: &str) -> Option<Entity> {
    world.get_resource::<OperationScene>()?.entity(id)
}

pub fn authored_id(world: &World, entity: Entity) -> Option<String> {
    world
        .get_resource::<OperationScene>()?
        .authored(world, entity)
        .map(|authored| authored.id)
}

#[derive(Component)]
pub struct SpawnedBy<B: 'static>(PhantomData<fn() -> B>);

impl<B: 'static> SpawnedBy<B> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<B: 'static> Default for SpawnedBy<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: 'static> Clone for SpawnedBy<B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: 'static> Copy for SpawnedBy<B> {}

impl<B: 'static> std::fmt::Debug for SpawnedBy<B> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "SpawnedBy<{}>", std::any::type_name::<B>())
    }
}

pub struct SceneNames<'w>(pub &'w World);

impl EntityIds for SceneNames<'_> {
    fn id(&self, entity: Entity) -> Option<String> {
        self.0
            .get::<SceneEntity>(entity)
            .map(|authored| authored.authored_id.clone())
    }
}
