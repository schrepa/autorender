use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_place::Error;
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
    pub streamed: bool,
}

impl OperationScene {
    pub fn entity(&self, world: &World, id: &str) -> Option<Entity> {
        self.entities.get(id).copied().or_else(|| {
            if !self.streamed {
                return None;
            }
            world
                .iter_entities()
                .find(|entity| {
                    entity.get::<SceneEntity>().is_some_and(|scene| {
                        scene.authored_id == id && scene.scene_id == self.id
                    })
                })
                .map(|entity| entity.id())
        })
    }

    pub fn authored(&self, world: &World, entity: Entity) -> Option<AuthoredId> {
        world.get::<SceneEntity>(entity).map(SceneEntity::authored)
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
