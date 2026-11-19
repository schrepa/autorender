use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_place::Error;
use serde_json::Value;

use crate::BrickId;

#[derive(Clone, Copy)]
pub struct SceneRead(pub fn(&mut World, Entity, &Value) -> Result<(), Error>);

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpawnedBy(pub BrickId);
