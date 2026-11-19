use bevy_ecs::component::ComponentId;
use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_core::{
    AssetLibrary, Drawn, GeometryHandle, GlobalTransforms, Kind, Lands, MaterialHandle, NameEntry,
    OperationScene, Parent, Reach, ReachMut, Refusals, Transform, scene_component,
    scene_components, unavailable,
};
use engine_scene::{SceneEntity, SceneInstance, SceneMaterial, TextureOrigin};
use engine_settings::{Error, ErrorCode, Expected};
use serde::Deserialize;
use serde_json::{Map, Value, json};

const SPAWNED_BY: &str = "SpawnedBy<";

pub fn row(world: &mut World) -> NameEntry {
    let kinds: Vec<_> = scene_components(world).cloned().collect();
    let mut covered: Vec<ComponentId> = kinds.iter().map(|kind| kind.register(world)).collect();
    covered.extend([
        world.register_component::<Transform>(),
        world.register_component::<Parent>(),
        world.register_component::<SceneEntity>(),
    ]);
    let listed = covered.clone();
    NameEntry::read("entities", Kind::State, move |reach, given| {
        let given: EntitiesArgs = crate::table::args(given)?;
        match given.entity {
            None => Ok(listing(reach, &listed, (given.offset, given.limit))),
            Some(id) => {
                let entity = find(reach, &id, "$.args.entity")?;
                Ok(one(reach, entity, &covered))
            }
        }
    })
    .args(
        engine_core::schema_of::<EntitiesArgs>(),
        json!({ "entity": "0v0", "detach": true }),
    )
    .set(Lands::ThisFrame, set)
    .pages()
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct EntitiesArgs {
    #[serde(default)]
    entity: Option<String>,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default)]
    detach: bool,
    #[serde(default)]
    component: Option<String>,
    #[serde(default)]
    value: Option<Value>,
}

fn find(reach: &Reach<'_>, id: &str, path: &str) -> Result<Entity, Error> {
    let world = reach.world;
    let authored = reach
        .part::<SceneInstance>()
        .and_then(|scene| scene.entity(id))
        .or_else(|| world.get_resource::<OperationScene>()?.entity(id));
    authored
        .or_else(|| {
            world
                .iter_entities()
                .map(|entity| entity.id())
                .find(|entity| entity.to_string() == id)
        })
        .filter(|&entity| world.get_entity(entity).is_ok())
        .ok_or_else(|| {
            Error::new(ErrorCode::UnknownEntity, path)
                .expected(Expected::Entity)
                .got(id)
        })
}

fn header(world: &World, entity: Entity) -> Map<String, Value> {
    let metadata = world.get::<SceneEntity>(entity);
    let mut row = Map::new();
    row.insert(
        "id".to_owned(),
        json!(metadata.map_or_else(|| entity.to_string(), |metadata| metadata.authored_id.clone())),
    );
    row.insert("entity".to_owned(), json!(entity.to_string()));
    if let Some(metadata) = metadata {
        row.insert("name".to_owned(), json!(metadata.name));
        row.insert("scene".to_owned(), json!(metadata.scene_id));
    }
    if let Some(by) = spawned_by(world, entity) {
        row.insert("by".to_owned(), json!(by));
    }
    row
}

fn spawned_by(world: &World, entity: Entity) -> Option<String> {
    world.inspect_entity(entity).ok()?.find_map(|info| {
        let name = info.name().shortname().to_string();
        name.strip_prefix(SPAWNED_BY)
            .and_then(|rest| rest.strip_suffix('>'))
            .map(str::to_owned)
    })
}

fn listing(
    reach: &Reach<'_>,
    covered: &[ComponentId],
    (offset, limit): (Option<usize>, Option<usize>),
) -> Value {
    let world = reach.world;
    let mut entities: Vec<Entity> = world.iter_entities().map(|entity| entity.id()).collect();
    entities.sort_unstable();
    let total = entities.len();
    let first = offset.unwrap_or(0);
    let rows: Vec<Value> = entities
        .into_iter()
        .skip(first)
        .take(limit.unwrap_or(usize::MAX))
        .map(|entity| {
            let mut row = header(world, entity);
            let names: Vec<String> = values(reach, entity, covered)
                .into_iter()
                .map(|(name, _)| name)
                .collect();
            row.insert("components".to_owned(), json!(names));
            Value::Object(row)
        })
        .collect();
    if offset.is_none() && limit.is_none() {
        return Value::Array(rows);
    }
    json!({ "total": total, "offset": first, "page": rows })
}

pub(crate) fn refused(reach: &Reach<'_>) -> Value {
    let world = reach.world;
    world
        .get_resource::<Refusals>()
        .into_iter()
        .flat_map(Refusals::iter)
        .map(|(entity, refused)| {
            let mut row = header(world, entity);
            row.insert("part".to_owned(), json!(refused.part));
            row.insert("error".to_owned(), json!(refused.reason));
            Value::Object(row)
        })
        .collect()
}

fn one(reach: &Reach<'_>, entity: Entity, covered: &[ComponentId]) -> Value {
    let mut row = header(reach.world, entity);
    if let Some(metadata) = reach.world.get::<SceneEntity>(entity) {
        row.insert("tags".to_owned(), json!(metadata.tags));
    }
    row.insert(
        "components".to_owned(),
        Value::Object(values(reach, entity, covered).into_iter().collect()),
    );
    Value::Object(row)
}

pub fn values(reach: &Reach<'_>, entity: Entity, covered: &[ComponentId]) -> Vec<(String, Value)> {
    let world = reach.world;
    let mut found: Vec<(String, Value)> = Vec::new();
    if let Some(transform) = world.get::<Transform>(entity) {
        found.push(("transform".to_owned(), transform_value(transform)));
        found.push(("global_transform".to_owned(), global_transform(world, entity)));
        found.push(("hierarchy".to_owned(), hierarchy(reach, entity)));
    }
    if let Some(drawn) = drawn_mesh(reach, entity) {
        found.push(("drawn_mesh".to_owned(), drawn));
    }
    for kind in scene_components(world) {
        if let Some(value) = kind.value(world, entity) {
            found.push((kind.key().to_owned(), value));
        }
    }
    if let Ok(infos) = world.inspect_entity(entity) {
        for info in infos.filter(|info| !covered.contains(&info.id())) {
            let short = info.name().shortname().to_string();
            if short.starts_with(SPAWNED_BY) {
                continue;
            }
            let reason = format!("{} has no serializer", info.name());
            found.push((short, unavailable(reason)));
        }
    }
    found
}

pub fn transform_value(transform: &Transform) -> Value {
    json!({
        "translation": transform.translation.to_array(),
        "rotation": transform.rotation.to_array(),
        "scale": transform.scale.to_array(),
    })
}

fn global_transform(world: &World, entity: Entity) -> Value {
    let global = world
        .get_resource::<GlobalTransforms>()
        .and_then(|globals| globals.get(entity));
    match global.and_then(|global| global.matrix()) {
        Some(matrix) => json!({
            "matrix": matrix.to_cols_array(),
            "translation": matrix.w_axis.truncate().to_array(),
        }),
        None => unavailable("the transform has not been propagated yet"),
    }
}

fn hierarchy(reach: &Reach<'_>, entity: Entity) -> Value {
    let world = reach.world;
    let parent = world.get::<Parent>(entity).map(Parent::parent);
    let parent_metadata = parent.and_then(|parent| world.get::<SceneEntity>(parent));
    let metadata = world.get::<SceneEntity>(entity);
    json!({
        "parent": parent.map(|parent| parent_metadata
            .map_or_else(|| parent.to_string(), |metadata| metadata.authored_id.clone())),
        "parent_alive": parent.map(|parent| world.get_entity(parent).is_ok()),
        "authored_parent": metadata.and_then(|metadata| metadata.authored_parent.clone()),
        "prefab_id": metadata.and_then(|metadata| metadata.prefab_id.clone()),
        "prefab_instance": metadata.and_then(|metadata| metadata.prefab_instance.clone()),
        "prefab_local_id": metadata.and_then(|metadata| metadata.prefab_local_id.clone()),
    })
}

fn drawn_mesh(reach: &Reach<'_>, entity: Entity) -> Option<Value> {
    let world = reach.world;
    let drawn = world.get::<Drawn>(entity)?;
    let handle: GeometryHandle = drawn.geometry?;
    let held: Option<MaterialHandle> = drawn.material;
    let assets = world.get_resource::<AssetLibrary>()?;
    let scene = reach.part::<SceneInstance>();
    let mesh = assets.mesh(handle).map(|mesh| {
        let origin = scene.and_then(|scene| scene.mesh_origin(handle));
        let bounds = mesh.bounds();
        json!({
            "id": origin.map(|(id, _)| id),
            "source": origin.map(|(_, source)| source),
            "vertices": mesh.vertices().len(),
            "indices": mesh.indices().len(),
            "local_bounds": { "min": bounds.min().to_array(), "max": bounds.max().to_array() },
            "colors": mesh.colors().is_some(),
            "uv1": mesh.uv1().is_some(),
        })
    });
    let material = held.and_then(|material| assets.material(material));
    let textures: Vec<Value> = material
        .into_iter()
        .flat_map(|material| material.maps().iter())
        .map(|(_, map)| {
            let origin = scene.and_then(|scene| scene.texture_origin(map.texture));
            let texture = assets.texture(map.texture);
            let uri = match origin.map(|(_, source)| source) {
                Some(TextureOrigin::File { uri }) => Some(uri),
                Some(TextureOrigin::Inline) | None => None,
            };
            json!({
                "id": origin.map(|(id, _)| id),
                "uri": uri,
                "inline": matches!(origin, Some((_, TextureOrigin::Inline))),
                "loaded": texture.is_some(),
                "width": texture.map(|texture| texture.width()),
                "height": texture.map(|texture| texture.height()),
            })
        })
        .collect();
    let material = material.map(|material| {
        let id = scene
            .and_then(|scene| scene.materials().find(|&(_, known)| Some(known) == held))
            .map(|(id, _)| id.to_owned());
        let texture = |handle| {
            scene
                .and_then(|scene| scene.texture_origin(handle))
                .map(|(id, _)| id.to_owned())
                .unwrap_or_default()
        };
        SceneMaterial::from_runtime(id.unwrap_or_default(), material, texture)
    });
    Some(json!({ "mesh": mesh, "material": material, "textures": textures }))
}

fn set(reach: &mut ReachMut<'_>, given: &Value) -> Result<Value, Error> {
    let given: EntitiesArgs = crate::table::args(given)?;
    let id = given.entity.ok_or_else(|| {
        Error::new(ErrorCode::MissingField, "$.args.entity").expected(Expected::Present)
    })?;
    let entity = reach.read(|reach| find(reach, &id, "$.args.entity"))?;
    match (given.parent, given.detach, given.component, given.value) {
        (parent, detach, None, None) if parent.is_some() != detach => {
            let parent = match parent {
                Some(parent) => Some(reach.read(|reach| find(reach, &parent, "$.args.parent"))?),
                None => None,
            };
            let before = reach.read(|reach| hierarchy(reach, entity));
            engine_core::set_parent(reach.world, entity, parent)
                .map_err(|error| error.within("$.args"))?;
            let after = reach.read(|reach| hierarchy(reach, entity));
            Ok(json!({ "before": before, "after": after }))
        }
        (None, false, Some(key), Some(value)) => {
            let world = &mut *reach.world;
            let kind = scene_component(world, &key).ok_or_else(|| {
                let keys: Vec<&str> = scene_components(world).map(|kind| kind.key()).collect();
                crate::table::unknown(&key, "$.args.component", &keys)
            })?;
            let before = kind.value(world, entity);
            kind.read(world, entity, &value)
                .map_err(|error| error.within("$.args.value"))?;
            let after = kind.value(world, entity);
            Ok(json!({ "before": before, "after": after }))
        }
        _ => Err(Error::new(ErrorCode::InvalidArguments, "$.args")
            .expected(Expected::Present)
            .variants(["parent", "detach", "component and value"])),
    }
}
