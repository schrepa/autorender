use std::collections::BTreeSet;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_core::{Error, ErrorCode, Expected, SceneContext, set_parents};
use engine_core::Call;
use engine_registry::{OperationEntry, OperationScene};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::hierarchy::{ExpandedObject, validate_object};
use crate::instance::spawn_entity;
use crate::validation::identifier;
use crate::{SceneKinds, SceneObject};

pub const SPAWN: &str = "spawn";
pub const REMOVE: &str = "remove";

pub fn spawn_operation() -> OperationEntry {
    OperationEntry::of(Call::exclusive::<SceneObject, Placed>(SPAWN, true, run_spawn))
}

pub fn remove_operation() -> OperationEntry {
    OperationEntry::of(Call::exclusive::<RemoveRequest, Placed>(REMOVE, true, run_remove))
}

#[derive(Serialize, schemars::JsonSchema)]
struct Placed {
    entity: String,
    node: u64,
}

fn scene_of(world: &World) -> Result<OperationScene, Error> {
    world
        .get_resource::<OperationScene>()
        .cloned()
        .ok_or_else(|| Error::new(ErrorCode::NoScene, "$").expected(Expected::Present))
}

pub fn spawn(world: &mut World, object: &SceneObject) -> Result<Entity, Error> {
    let mut scene = scene_of(world)?;
    identifier(&object.id, "$.id")?;
    if scene.entity(&object.id).is_some() {
        return Err(Error::new(ErrorCode::DuplicateId, "$.id")
            .expected(Expected::Unique)
            .got(&object.id));
    }
    let kinds = SceneKinds::of(world);
    let transform = validate_object(object, "$", &BTreeSet::new(), &kinds)?;
    kinds.check_placement(object, "$")?;
    let mut skipped = Vec::new();
    kinds.unlisted(object, "$", &mut skipped);
    if let Some(first) = skipped.into_iter().next() {
        return Err(first);
    }
    let parent = object
        .parent
        .as_deref()
        .map(|parent| {
            scene.entity(parent).ok_or_else(|| {
                Error::new(ErrorCode::UnknownParent, "$.parent").got(parent)
            })
        })
        .transpose()?;
    let expanded = ExpandedObject {
        source: object.clone(),
        path: "$".to_owned(),
        prefab_id: None,
        prefab_instance: None,
        prefab_local_id: None,
    };
    let entity = spawn_entity(world, &scene.id, &expanded, transform);
    let context = world.get_resource::<SceneContext>().cloned();
    let landed = crate::loading::within(context.as_ref(), || {
        let cx = kinds.cx(context.as_ref(), None);
        kinds.read(world, entity, object, "$", &cx)?;
        match parent {
            Some(parent) => set_parents(world, &[(entity, Some(parent))]),
            None => Ok(()),
        }
    });
    if let Err(error) = landed {
        let _ = world.try_despawn(entity);
        return Err(error);
    }
    scene.installed.insert(object.id.clone(), entity);
    world.insert_resource(scene);
    Ok(entity)
}

pub fn remove(world: &mut World, id: &str) -> Result<Entity, Error> {
    let mut scene = scene_of(world)?;
    let entity = scene
        .installed
        .remove(id)
        .ok_or_else(|| Error::new(ErrorCode::UnknownEntity, "$.entity").got(id))?;
    world.insert_resource(scene);
    world.despawn(entity);
    Ok(entity)
}

fn run_spawn(world: &mut World, request: Value) -> Result<Value, Error> {
    let object: SceneObject = engine_settings::parse_value(request)?;
    let entity = spawn(world, &object)?;
    Ok(json!(Placed {
        entity: object.id,
        node: entity.to_bits(),
    }))
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct RemoveRequest {
    entity: String,
}

fn run_remove(world: &mut World, request: Value) -> Result<Value, Error> {
    let RemoveRequest { entity: id } = engine_settings::parse_value(request)?;
    let entity = remove(world, &id)?;
    Ok(json!(Placed {
        entity: id,
        node: entity.to_bits(),
    }))
}
