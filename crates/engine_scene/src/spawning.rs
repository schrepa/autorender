use std::collections::BTreeSet;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_core::{Error, ErrorCode, Expected, OperationScene, SceneContext, set_parents};
use serde_json::Value;

use crate::hierarchy::{ExpandedObject, validate_object};
use crate::instance::spawn_entity;
use crate::validation::identifier;
use crate::{SceneKinds, SceneObject};

fn no_scene() -> Error {
    Error::new(ErrorCode::NoScene, "$").expected(Expected::Present)
}

fn scene_of(world: &World) -> Result<&OperationScene, Error> {
    world.get_resource::<OperationScene>().ok_or_else(no_scene)
}

pub fn spawn(world: &mut World, object: &SceneObject) -> Result<Entity, Error> {
    let scene = scene_of(world)?;
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
            scene.named_in(world, parent, "$.parent").map_err(|error| {
                if error.code == ErrorCode::UnknownEntity {
                    Error::new(ErrorCode::UnknownParent, "$.parent").got(parent)
                } else {
                    error
                }
            })
        })
        .transpose()?;
    let scene_id = scene.id.clone();
    let expanded = ExpandedObject {
        source: object.clone(),
        path: "$".to_owned(),
        prefab_id: None,
        prefab_instance: None,
        prefab_local_id: None,
    };
    let entity = spawn_entity(world, &scene_id, &expanded, transform);
    let context = world.get_resource::<SceneContext>().cloned();
    let landed = crate::loading::within(context.as_ref(), || {
        let cx = kinds.cx(context.as_ref(), None);
        kinds.read(world, entity, object, "$", &cx)?;
        if let Some(parent) = parent {
            set_parents(world, &[(entity, Some(parent))])?;
        }
        engine_core::admit(world, entity, None, "$")
    });
    if let Err(error) = landed {
        let _ = world.try_despawn(entity);
        if let Some(mut scene) = world.get_resource_mut::<OperationScene>() {
            scene.forget(&object.id);
        }
        return Err(error);
    }
    Ok(entity)
}

pub fn remove(world: &mut World, id: &str) -> Result<usize, Error> {
    let entity = scene_of(world)?.named_in(world, id, "$")?;
    let gone = engine_core::subtree(world, entity);
    for &held in &gone {
        let _ = world.try_despawn(held);
    }
    Ok(gone.len())
}

fn unused_id(world: &World) -> Result<String, Error> {
    let scene = scene_of(world)?;
    let mut at = scene.entity_count();
    loop {
        let id = format!("spawned-{at}");
        if scene.entity(&id).is_none() {
            return Ok(id);
        }
        at += 1;
    }
}

pub fn described(world: &World, mut described: Value) -> Result<SceneObject, Error> {
    let Value::Object(fields) = &mut described else {
        return Err(Error::new(ErrorCode::InvalidType, "$")
            .expected(Expected::Object)
            .got(described));
    };
    if !fields.contains_key("id") {
        fields.insert("id".to_owned(), Value::String(unused_id(world)?));
    }
    engine_settings::parse_value(described)
}
