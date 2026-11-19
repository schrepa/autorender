use std::collections::BTreeSet;
use std::sync::Arc;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_core::{Error, ErrorCode, Expected, OperationScene, SceneContext, SceneCx, set_parents};
use serde_json::Value;

use crate::hierarchy::{ExpandedObject, validate_object};
use crate::instance::spawn_entity;
use crate::validation::identifier;
use crate::{SceneKinds, SceneObject};

pub fn spawn(world: &mut World, object: &SceneObject) -> Result<Entity, Error> {
    if !world.contains_resource::<OperationScene>() {
        world.insert_resource(OperationScene::new(String::new(), Arc::default()));
    }
    let context = world.get_resource::<SceneContext>().cloned();
    let assets = context
        .as_ref()
        .map_or_else(Arc::default, |context| Arc::clone(&context.assets));
    let kinds = SceneKinds::of(world);
    let scene = world.resource::<OperationScene>();
    identifier(&object.id, "$.id")?;
    if scene.entity(&object.id).is_some() {
        return Err(Error::new(ErrorCode::DuplicateId, "$.id")
            .expected(Expected::Unique)
            .got(&object.id));
    }
    let transform = assets.scope(|| {
        let transform = validate_object(object, "$", &BTreeSet::new(), &kinds)?;
        let mut skipped = Vec::new();
        kinds.unlisted(object, "$", &mut skipped);
        skipped.into_iter().next().map_or(Ok(transform), Err)
    })?;
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
    let landed = assets.scope(|| {
        let cx = SceneCx::built(context.as_ref());
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
    let entity = match world.get_resource::<OperationScene>() {
        Some(scene) => scene.named_in(world, id, "$")?,
        None => return Err(Error::new(ErrorCode::UnknownEntity, "$").got(id)),
    };
    let gone = engine_core::subtree(world, entity);
    for &held in &gone {
        let _ = world.try_despawn(held);
    }
    Ok(gone.len())
}

fn unused_id(world: &World) -> String {
    let scene = world.get_resource::<OperationScene>();
    let mut at = scene.map_or(0, OperationScene::entity_count);
    loop {
        let id = format!("spawned-{at}");
        if scene.is_none_or(|scene| scene.entity(&id).is_none()) {
            return id;
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
        fields.insert("id".to_owned(), Value::String(unused_id(world)));
    }
    engine_settings::parse_value(described)
}
