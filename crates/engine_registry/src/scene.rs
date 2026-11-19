use std::marker::PhantomData;

use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_place::Error;
use serde_json::Value;

#[derive(Clone, Copy)]
pub struct SceneRead(pub fn(&mut World, Entity, &Value) -> Result<(), Error>);

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
