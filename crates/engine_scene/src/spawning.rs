use std::collections::BTreeSet;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_core::{
    Callable, Error, ErrorCode, Expected, OperationScene, Rows, SceneContext, set_parents,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::hierarchy::{ExpandedObject, validate_object};
use crate::instance::spawn_entity;
use crate::validation::identifier;
use crate::{SceneKinds, SceneObject};

pub struct SpawnOperation;

impl Callable for SpawnOperation {
    const NAME: &'static str = "spawn";
    type Request = SpawnRequest;
    type Reply = Rows<SpawnedEntity>;

    fn answer(world: &mut World, request: SpawnRequest) -> Result<Self::Reply, Error> {
        Ok(Rows::each("entities", request.entities, |description| {
            let object = described(world, description)?;
            spawn(world, &object)?;
            Ok(SpawnedEntity { entity: object.id })
        }))
    }

    fn request_schema(world: &World) -> Value {
        spawn_request(world)
    }
}

pub struct RemoveOperation;

impl Callable for RemoveOperation {
    const NAME: &'static str = "remove";
    type Request = RemoveRequest;
    type Reply = Rows<RemovedEntity>;

    fn answer(world: &mut World, request: RemoveRequest) -> Result<Self::Reply, Error> {
        Ok(Rows::each("entities", request.entities, |id| {
            let removed = remove(world, &id)?;
            Ok(RemovedEntity {
                entity: id,
                removed,
            })
        }))
    }
}

fn spawn_request(world: &World) -> Value {
    let mut schema = engine_core::schema_of::<SpawnRequest>();
    if let Some(object) = schema
        .get_mut("$defs")
        .and_then(|definitions| definitions.get_mut(SPAWNED))
        .and_then(Value::as_object_mut)
    {
        if let Some(properties) = object.get_mut("properties").and_then(Value::as_object_mut) {
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

#[derive(Serialize, schemars::JsonSchema)]
pub struct SpawnedEntity {
    pub entity: String,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct RemovedEntity {
    pub entity: String,
    pub removed: usize,
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
pub struct SpawnRequest {
    #[schemars(with = "Vec<Described>")]
    pub entities: Vec<Value>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RemoveRequest {
    pub entities: Vec<String>,
}

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
