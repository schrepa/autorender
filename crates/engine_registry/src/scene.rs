use std::marker::PhantomData;

use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_place::{Error, OperationScene};
use serde_json::Value;

#[derive(Clone, Copy)]
pub struct SceneRead(pub fn(&mut World, Entity, &Value) -> Result<(), Error>);

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
