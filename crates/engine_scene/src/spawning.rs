use std::collections::BTreeSet;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_core::{Call, Lands, Rows};
use engine_core::{Error, ErrorCode, Expected, SceneContext, set_parents};
use engine_registry::{OperationEntry, OperationScene};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::hierarchy::{ExpandedObject, validate_object};
use crate::instance::spawn_entity;
use crate::validation::identifier;
use crate::{SceneKinds, SceneObject};

pub const SPAWN: &str = "spawn";
pub const REMOVE: &str = "remove";

pub fn spawn_operation() -> OperationEntry {
    OperationEntry::of(
        Call::exclusive::<SpawnRequest, Rows<Placed>>(SPAWN, (true, Lands::Now), run_spawn)
            .request_in(spawn_request),
    )
}

fn spawn_request(world: &World) -> Value {
    let mut schema = engine_core::schema_of::<SpawnRequest>();
    if let Some(object) = schema
        .get_mut("$defs")
        .and_then(|definitions| definitions.get_mut(SPAWNED))
        .and_then(Value::as_object_mut)
    {
        if let Some(properties) = object
            .get_mut("properties")
            .and_then(Value::as_object_mut)
        {
            for kind in engine_core::scene_components(world) {
                properties.insert(
                    kind.key().to_owned(),
                    engine_core::authored_component(kind.key()),
                );
            }
        }
        object.insert("additionalProperties".to_owned(), Value::Bool(false));
    }
    schema
}

pub fn remove_operation() -> OperationEntry {
    OperationEntry::of(Call::exclusive::<RemoveRequest, Rows<Placed>>(
        REMOVE,
        (true, Lands::Now),
        run_remove,
    ))
}

#[derive(Serialize, schemars::JsonSchema)]
struct Placed {
    entity: String,
}

struct Described;

const SPAWNED: &str = "SpawnedObject";

impl schemars::JsonSchema for Described {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        SPAWNED.into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let mut schema = <SceneObject as schemars::JsonSchema>::json_schema(generator);
        if let Some(Value::Array(required)) = schema.get_mut("required") {
            required.retain(|field| field != "id");
        }
        schema
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct SpawnRequest {
    #[schemars(with = "Vec<Described>")]
    entities: Vec<Value>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct RemoveRequest {
    entities: Vec<String>,
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
            scene
                .entity(parent)
                .ok_or_else(|| Error::new(ErrorCode::UnknownParent, "$.parent").got(parent))
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
        .ok_or_else(|| Error::new(ErrorCode::UnknownEntity, "$").got(id))?;
    world.insert_resource(scene);
    let _ = world.try_despawn(entity);
    Ok(entity)
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

fn described(world: &World, mut described: Value) -> Result<SceneObject, Error> {
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

fn finished<T: Serialize>(rows: &Rows<T>) -> Result<Value, Error> {
    serde_json::to_value(rows)
        .map_err(|error| Error::new(ErrorCode::InvalidValue, "$").got(error.to_string()))
}

fn run_spawn(world: &mut World, request: Value) -> Result<Value, Error> {
    let SpawnRequest { entities } = engine_settings::parse_value(request)?;
    let rows = Rows::each("entities", entities, |description| {
        let object = described(world, description)?;
        spawn(world, &object)?;
        Ok(Placed { entity: object.id })
    });
    finished(&rows)
}

fn run_remove(world: &mut World, request: Value) -> Result<Value, Error> {
    let RemoveRequest { entities } = engine_settings::parse_value(request)?;
    let rows = Rows::each("entities", entities, |id| {
        remove(world, &id)?;
        Ok(Placed { entity: id })
    });
    finished(&rows)
}
