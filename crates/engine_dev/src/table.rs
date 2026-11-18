use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_core::{
    AssetLibrary, Camera, Drawn, FixedTime, GlobalTransforms, Held, Kind, Lands, NameEntry, Pause,
    Projection, Reach, ReachMut, Transform, glam::Vec3, main_camera, place_camera,
};
use engine_registry::{BrickStates, Registry, SettingsEntry, StatsEntry};
use engine_scene::{SceneEntity, SceneInstance};
use engine_settings::{Error, ErrorCode, Expected};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::request::Verb;

pub fn args<T: serde::de::DeserializeOwned>(args: &Value) -> Result<T, Error> {
    engine_settings::parse_value_at(args.clone(), "$.args")
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct SectionArgs {
    #[serde(default)]
    value: Option<Value>,
    #[serde(default)]
    replace: Option<Value>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct CameraArgs {
    #[serde(default)]
    position: Option<[f32; 3]>,
    #[serde(default)]
    target: Option<[f32; 3]>,
    #[serde(default)]
    vertical_fov_radians: Option<f32>,
    #[serde(default)]
    near: Option<f32>,
    #[serde(default)]
    far: Option<f32>,
    #[serde(default)]
    release: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct PauseArgs {
    value: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct SpawnArgs {
    entities: Vec<Value>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct RemoveArgs {
    entities: Vec<String>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct NoArgs {}

pub fn install(world: &mut World) {
    let Some(registry) = world.get_resource::<Registry>() else {
        return;
    };
    let mut rows: Vec<NameEntry> = registry.sections().map(setting).collect();
    rows.extend(registry.stats().map(stats));
    rows.extend([
        bare("status", |reach| Ok(status(reach))),
        bare("assets", assets),
        bare("bricks", |reach| Ok(bricks(reach.world))),
        bare("entities.refused", |reach| Ok(crate::entities::refused(reach))),
        bare("kept", kept),
        bare("records", |reach| {
            Ok(json!(reach.world.get_resource::<engine_core::Records>()))
        }),
        camera(),
        pause(),
        NameEntry::call("spawn", spawn).args(
            engine_core::schema_of::<SpawnArgs>(),
            json!({ "entities": [{ "transform": { "translation": [0.0, 1.0, 0.0] } }] }),
        ),
        NameEntry::call("remove", remove).args(
            engine_core::schema_of::<RemoveArgs>(),
            json!({ "entities": ["spawned-0"] }),
        ),
        NameEntry::call("digest", |reach, _| {
            Ok(engine_core::Answer::Value(reach.read(crate::digest::digest)))
        }),
    ]);
    rows.push(crate::entities::row(world));
    for row in rows {
        engine_registry::register(world, row);
    }
}

fn none(given: &Value) -> Result<(), Error> {
    args::<NoArgs>(given).map(drop)
}

pub(crate) fn shapeless(entry: &NameEntry) -> bool {
    entry.args.as_object().is_some_and(Map::is_empty)
}

fn bare(
    name: &'static str,
    read: impl Fn(&Reach<'_>) -> Result<Value, Error> + Send + Sync + 'static,
) -> NameEntry {
    NameEntry::read(name, Kind::State, move |reach, _| read(reach))
}

fn setting(section: &SettingsEntry) -> NameEntry {
    let section = *section;
    NameEntry::read(format!("settings.{}", section.key), Kind::Setting, move |reach, given| {
        none(given)?;
        in_scene(reach.world, || section.current(reach.world))
    })
    .args(
        engine_core::schema_of::<SectionArgs>(),
        json!({ "value": section.default_value() }),
    )
    .set(Lands::ThisFrame, move |reach, given| {
        let given: SectionArgs = args(given)?;
        let before = in_scene(reach.world, || section.current(reach.world))?;
        let after = match (given.value, given.replace) {
            (Some(patch), None) => {
                let mut merged = before.clone();
                overlay(&mut merged, patch);
                merged
            }
            (None, Some(whole)) => whole,
            (Some(_), Some(_)) | (None, None) => {
                return Err(Error::new(ErrorCode::InvalidArguments, "$.args")
                    .expected(Expected::Present)
                    .variants(["value", "replace"]));
            }
        };
        section.set_at(reach.world, &after, "$.args")?;
        let after = in_scene(reach.world, || section.current(reach.world))?;
        Ok(json!({ "before": before, "after": after }))
    })
    .owned_by(section.brick)
}

fn stats(stats: &StatsEntry) -> NameEntry {
    let stats = *stats;
    NameEntry::read(format!("stats.{}", stats.key), Kind::State, move |reach, _| {
        Ok(stats.read(reach.world))
    })
    .owned_by(stats.brick)
}

fn in_scene<R>(world: &World, run: impl FnOnce() -> R) -> R {
    match world.get_resource::<engine_core::SceneContext>() {
        Some(context) => context.scope(run),
        None => run(),
    }
}

fn overlay(base: &mut Value, patch: Value) {
    match (base, patch) {
        (Value::Object(base), Value::Object(patch)) => {
            for (key, value) in patch {
                overlay(base.entry(key).or_insert(Value::Null), value);
            }
        }
        (base, patch) => *base = patch,
    }
}

fn status(reach: &Reach<'_>) -> Value {
    let world = reach.world;
    let registry = world.get_resource::<Registry>();
    json!({
        "entities": world.entities().count_spawned(),
        "scene": reach.part::<SceneInstance>().map(|scene| json!({
            "id": scene.id(),
            "revision": scene.revision(),
            "authored_entities": scene.entity_count(),
        })),
        "bricks": registry.map(|registry| registry.names().collect::<Vec<_>>()),
        "notices": registry.map(Registry::notices),
        "fixed_delta_seconds": world.resource::<FixedTime>().delta_seconds,
    })
}

fn kept(reach: &Reach<'_>) -> Result<Value, Error> {
    let kept = reach.world.get_resource::<engine_core::Kept>().ok_or_else(|| {
        Error::new(ErrorCode::MissingResource, "$.kept").expected(Expected::Present)
    })?;
    let listed = kept.listing()?;
    Ok(json!({
        "place": kept.place(),
        "kept": listed
            .into_iter()
            .map(|(name, bytes)| json!({ "name": name, "bytes": bytes }))
            .collect::<Vec<_>>(),
    }))
}

fn bricks(world: &World) -> Value {
    let (Some(registry), Some(states)) = (
        world.get_resource::<Registry>(),
        world.get_resource::<BrickStates>(),
    ) else {
        return json!({});
    };
    states
        .iter()
        .filter_map(|(brick, on)| {
            let record = registry.record(brick)?;
            Some((
                record.package.to_owned(),
                json!({ "on": on, "switch": record.switch_key() }),
            ))
        })
        .collect::<Map<String, Value>>()
        .into()
}

fn assets(reach: &Reach<'_>) -> Result<Value, Error> {
    let world = reach.world;
    let assets = world.get_resource::<AssetLibrary>().ok_or_else(|| {
        Error::new(ErrorCode::MissingResource, "$.assets")
            .expected(Expected::Present)
            .got("AssetLibrary")
    })?;
    let textures: Vec<Value> = reach
        .part::<SceneInstance>()
        .into_iter()
        .flat_map(SceneInstance::textures)
        .map(|(id, handle)| {
            let texture = assets.texture(handle);
            json!({
                "id": id,
                "loaded": texture.is_some(),
                "width": texture.map(|texture| texture.width()),
                "height": texture.map(|texture| texture.height()),
                "bytes": texture.map(|texture| texture.byte_len()),
            })
        })
        .collect();
    let mut missing: Vec<(Entity, &str)> = Vec::new();
    if let Some(mut drawn) = world.try_query::<(Entity, &Drawn)>() {
        for (entity, drawn) in drawn.iter(world) {
            if let Some(mesh) = drawn.geometry
                && let Some(what) = missing_of(assets, mesh, drawn.material)
            {
                missing.push((entity, what));
            }
        }
    }
    missing.sort_unstable_by_key(|&(entity, _)| entity);
    let transforms = world.get_resource::<GlobalTransforms>();
    let missing: Vec<Value> = missing
        .into_iter()
        .map(|(entity, what)| {
            let metadata = world.get::<SceneEntity>(entity);
            json!({
                "entity": entity.to_string(),
                "id": metadata.map(|metadata| &metadata.authored_id),
                "missing": what,
                "position": transforms
                    .and_then(|transforms| transforms.get(entity))
                    .and_then(|global| global.translation())
                    .map(|translation| translation.to_array()),
            })
        })
        .collect();
    Ok(json!({
        "meshes": assets.mesh_count(),
        "materials": assets.material_count(),
        "textures": assets.texture_count(),
        "texture_cpu_bytes": assets.texture_bytes(),
        "resident_cpu_bytes": assets.resident_bytes(),
        "scene_textures": textures,
        "missing": missing,
    }))
}

fn missing_of(
    assets: &AssetLibrary,
    mesh: engine_core::GeometryHandle,
    material: Option<engine_core::MaterialHandle>,
) -> Option<&'static str> {
    match material.map(|material| assets.material(material)) {
        Some(None) => Some("material"),
        None | Some(Some(_)) if assets.mesh(mesh).is_none() => Some("mesh"),
        None => None,
        Some(Some(material)) => material
            .textures()
            .any(|texture| assets.texture(texture).is_none())
            .then_some("texture"),
    }
}

fn camera() -> NameEntry {
    NameEntry::read("camera", Kind::State, |reach, given| {
        none(given)?;
        let world = reach.world;
        let held = world.get_resource::<Held>().and_then(Held::camera).is_some();
        Ok(match main_camera(world) {
            Some(camera) => json!({
                "entity": camera.to_string(),
                "transform": world.get::<Transform>(camera).map(crate::entities::transform_value),
                "lens": world.get::<Camera>(camera),
                "held": held,
            }),
            None => engine_core::unavailable("no_main_camera"),
        })
    })
    .args(
        engine_core::schema_of::<CameraArgs>(),
        json!({
            "position": [0.0, 2.0, 8.0],
            "target": [0.0, 0.0, 0.0],
            "vertical_fov_radians": 0.9,
            "near": 0.1,
            "far": 500.0,
        }),
    )
    .set(Lands::ThisFrame, set_camera)
}

fn set_camera(reach: &mut ReachMut<'_>, given: &Value) -> Result<Value, Error> {
    let given: CameraArgs = args(given)?;
    let world = &mut *reach.world;
    let camera = main_camera(world)
        .ok_or_else(|| Error::new(ErrorCode::InvalidCamera, "$").expected(Expected::Present))?;
    let before = world.get::<Transform>(camera).map(crate::entities::transform_value);
    if given.release {
        world.resource_mut::<Held>().hold_camera(None);
        return Ok(json!({ "before": before, "after": before, "held": false }));
    }
    fn needed<T>(value: Option<T>, key: &str) -> Result<T, Error> {
        value.ok_or_else(|| {
            Error::new(ErrorCode::MissingField, format!("$.args.{key}")).expected(Expected::Present)
        })
    }
    let position = Vec3::from_array(needed(given.position, "position")?);
    let target = Vec3::from_array(needed(given.target, "target")?);
    let lens = Camera {
        near: needed(given.near, "near")?,
        far: needed(given.far, "far")?,
        projection: Projection::Perspective {
            vertical_fov_radians: needed(given.vertical_fov_radians, "vertical_fov_radians")?,
        },
    };
    lens.check().map_err(|error| error.within("$.args"))?;
    if !position.is_finite()
        || !target.is_finite()
        || (target - position).length_squared() < Camera::MIN_DEPTH * Camera::MIN_DEPTH
    {
        return Err(Error::new(ErrorCode::InvalidCamera, "$.args.target")
            .expected(Expected::Finite)
            .got(json!({ "position": position.to_array(), "target": target.to_array() })));
    }
    let at = Transform::looking_at(position, target);
    place_camera(world, camera, at, lens);
    world.resource_mut::<Held>().hold_camera(Some((at, lens)));
    let after = world.get::<Transform>(camera).map(crate::entities::transform_value);
    Ok(json!({ "before": before, "after": after, "held": true }))
}

fn pause() -> NameEntry {
    NameEntry::read("pause", Kind::State, |reach, given| {
        none(given)?;
        let world = reach.world;
        Ok(json!({
            "paused": world.resource::<Held>().paused,
            "game": world.get_resource::<Pause>().map(|pause| pause.reasons().collect::<Vec<_>>()),
        }))
    })
    .args(engine_core::schema_of::<PauseArgs>(), json!({ "value": true }))
    .set(Lands::NextFrame, |reach, given| {
        let paused = args::<PauseArgs>(given)?.value;
        let mut held = reach.world.resource_mut::<Held>();
        let before = held.paused;
        held.paused = paused;
        Ok(json!({ "before": before, "after": paused }))
    })
}

fn each<T>(
    entities: Vec<T>,
    mut run: impl FnMut(T) -> Result<Value, Error>,
) -> engine_core::Answer {
    let rows: Vec<Value> = entities
        .into_iter()
        .enumerate()
        .map(|(at, entity)| match run(entity) {
            Ok(done) => done,
            Err(error) => json!({ "refused": error.within(&format!("$.args.entities[{at}]")) }),
        })
        .collect();
    engine_core::Answer::Value(Value::Array(rows))
}

fn spawn(reach: &mut ReachMut<'_>, given: &Value) -> Result<engine_core::Answer, Error> {
    let world = &mut *reach.world;
    let given: SpawnArgs = args(given)?;
    Ok(each(given.entities, |described| {
        let object = engine_scene::described(world, described)?;
        let entity = engine_scene::spawn(world, &object)?;
        Ok(json!({ "id": object.id, "entity": entity.to_string() }))
    }))
}

fn remove(reach: &mut ReachMut<'_>, given: &Value) -> Result<engine_core::Answer, Error> {
    let world = &mut *reach.world;
    let given: RemoveArgs = args(given)?;
    Ok(each(given.entities, |id| {
        let removed = engine_scene::remove(world, &id)?;
        Ok(json!({ "id": id, "removed": removed }))
    }))
}

pub fn verbs(entry: &NameEntry) -> Vec<&'static str> {
    [
        (entry.readable(), Verb::Read),
        (entry.settable(), Verb::Set),
        (entry.callable(), Verb::Call),
    ]
    .into_iter()
    .filter(|(has, _)| *has)
    .map(|(_, verb)| verb.name())
    .collect()
}

fn listed(entry: &NameEntry) -> Value {
    let mut row = json!({
        "name": entry.name,
        "kind": entry.kind,
        "owner": entry.owner,
        "verbs": verbs(entry),
    });
    if let Some(unit) = entry.unit {
        row["unit"] = json!(unit);
    }
    if let Some(lands) = entry.lands {
        row["lands"] = json!(lands);
    }
    if let Some(missing) = &entry.missing {
        row["unavailable"] = json!(missing);
    }
    if let Some(changes) = entry.changes_frame {
        row["changes_frame"] = json!(changes);
    }
    row
}

fn described(entry: &NameEntry) -> Value {
    let mut row = listed(entry);
    row["args"] = if shapeless(entry) {
        engine_core::schema_of::<NoArgs>()
    } else {
        entry.args.clone()
    };
    row["examples"] = verbs(entry)
        .into_iter()
        .map(|verb| {
            let args = if verb == Verb::Read.name() {
                json!({})
            } else {
                entry.example.clone()
            };
            json!({ "verb": verb, "name": entry.name, "args": args })
        })
        .collect();
    row
}

pub fn describe(reach: &Reach<'_>, name: &str) -> Result<Value, Error> {
    let registry = reach.world.get_resource::<Registry>().ok_or_else(|| {
        Error::new(ErrorCode::MissingResource, "$.name")
            .expected(Expected::Present)
            .got("Registry")
    })?;
    if let Some(entry) = registry.entry(name) {
        return Ok(described(entry));
    }
    let prefix = format!("{name}.");
    let mut rows: Vec<Value> = registry
        .table()
        .filter(|entry| name.is_empty() || entry.name.starts_with(&prefix))
        .map(|entry| listed(entry))
        .collect();
    if rows.is_empty() {
        let names: Vec<&str> = registry.table().map(|entry| &*entry.name).collect();
        return Err(unknown(name, "$.name", &names));
    }
    rows.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    Ok(json!({ "verbs": Verb::names().collect::<Vec<_>>(), "names": rows }))
}

pub fn unknown(name: &str, path: &str, known: &[&str]) -> Error {
    let mut ranked: Vec<(usize, &str)> = known
        .iter()
        .map(|candidate| (distance(name, candidate), *candidate))
        .collect();
    ranked.sort_unstable();
    let nearest: Vec<&str> = ranked.iter().take(5).map(|&(_, candidate)| candidate).collect();
    let error = Error::new(ErrorCode::UnknownVariant, path)
        .got(name)
        .variants(nearest.iter().copied());
    match nearest.first() {
        Some(best) => error.nearest(*best),
        None => error,
    }
}

fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, left) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, &right) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (above + 1)
                .min(row[j] + 1)
                .min(diagonal + usize::from(left != right));
            diagonal = above;
        }
    }
    row[b.len()]
}
