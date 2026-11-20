use std::collections::BTreeSet;
use std::sync::Arc;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_core::{
    Assets, Error, ErrorCode, Expected, OperationScene, SceneContext, SceneCx, set_parents,
};
use serde_json::Value;

use crate::hierarchy::{ExpandedObject, validate_object};
use crate::validation::identifier;
use crate::{SceneKinds, SceneObject};

pub fn named<R>(world: &mut World, run: impl FnOnce(&mut World) -> R) -> R {
    match world.get_resource::<Assets>().map(Assets::names) {
        Some(names) => names.scope(|| run(world)),
        None => run(world),
    }
}

pub fn spawn(world: &mut World, object: &SceneObject) -> Result<Entity, Error> {
    if !world.contains_resource::<OperationScene>() {
        world.insert_resource(OperationScene::new(String::new(), Arc::default()));
    }
    let context = world.get_resource::<SceneContext>().cloned();
    let kinds = SceneKinds::of(world);
    let scene = world.resource::<OperationScene>();
    identifier(&object.id, "$.id")?;
    if scene.entity(&object.id).is_some() {
        return Err(Error::new(ErrorCode::DuplicateId, "$.id")
            .expected(Expected::Unique)
            .got(&object.id));
    }
    let transform = validate_object(object, "$", &BTreeSet::new())?;
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
    let expanded = ExpandedObject {
        source: object.clone(),
        path: "$".to_owned(),
        prefab_id: None,
        prefab_instance: None,
        prefab_local_id: None,
    };
    let authored = expanded.scene_entity(&scene.id);
    let entity = world.spawn((transform, authored)).id();
    let landed = named(world, |world| {
        let cx = SceneCx::new(context.as_ref());
        kinds.read(world, entity, object, "$", &cx)?;
        match parent {
            Some(parent) => set_parents(world, &[(entity, Some(parent))]),
            None => Ok(()),
        }
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
    engine_settings::parse_value(&described)
}
