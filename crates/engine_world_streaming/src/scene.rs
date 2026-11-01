use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::prelude::{Entity, Resource};
use engine_codec::closest;
use engine_core::__private::schemars::{JsonSchema, Schema};
use engine_core::{
    ComponentKind, Error, ErrorCode, Expected, InvalidSetting, MAX_HIERARCHY_DEPTH as MAX_DEPTH,
    MAX_ID_BYTES, Placed, Placement, Rule, SceneContext, ScenePath, Section, Setting, Transform,
    fields, is_id, parse_value_at, state, valid_global_matrix,
};
use engine_places::ground::Grounded;
use glam::{EulerRot, Mat4, Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const MAX_ALONG: usize = 10_000;

const MAX_DISTANCE: f32 = 1.0e7;

const MIN_EVERY: f32 = 0.01;

const MAX_OFFSET: f32 = 1.0e4;

const MAX_REACH: f32 = 1.0e6;

const MAX_NAME_BYTES: usize = 128;

const TAGS: Rule = Rule::count(0.0, 16.0);

const GROUNDED: &str = "grounded";

#[derive(Clone, Debug, Default, PartialEq, Resource, Serialize, Deserialize, JsonSchema)]
#[component(immutable)]
#[serde(default, deny_unknown_fields)]
#[schemars(crate = "engine_core::__private::schemars", rename = "SceneStreamed")]
pub struct StreamedScene {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub streamed: Vec<StreamedObject>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub along: Vec<StreamedAlong>,
}

impl Section for StreamedScene {
    const KEY: &'static str = "streamed";

    fn check_scene(&self, scene: &SceneContext) -> Result<(), Error> {
        self.expand(scene).map(drop)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(
    crate = "engine_core::__private::schemars",
    rename = "SceneObject",
    transform = object_rules
)]
pub struct StreamedObject {
    #[schemars(transform = engine_core::stated(engine_core::fields::id()))]
    pub id: String,
    #[serde(default)]
    #[schemars(length(max = MAX_NAME_BYTES))]
    pub name: String,
    #[schemars(transform = engine_core::stated(engine_core::fields::id()))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<String>,
    #[schemars(transform = engine_core::stated(engine_core::fields::id()))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(transform = engine_core::stated(engine_core::fields::id()))]
    pub parent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(transform = engine_core::stated(engine_core::fields::id()))]
    pub prefab: Option<String>,
    #[serde(flatten)]
    #[schemars(skip)]
    pub components: BTreeMap<String, Value>,
    #[serde(default)]
    pub transform: StreamedTransform,
    #[serde(default)]
    pub tags: Vec<String>,
}

fn object_rules(schema: &mut Schema) {
    state(schema, "tags", TAGS);
}

impl StreamedObject {
    fn validate(&self) -> Result<(), InvalidSetting> {
        Setting::check(&self.transform, "transform", Rule::Any)?;
        Setting::check(&self.tags, "tags", TAGS)
    }

    pub(crate) fn authored(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.components
            .iter()
            .filter(|(_, value)| !value.is_null())
            .map(|(name, value)| (name.as_str(), value))
    }
}

fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    #[schemars(crate = "engine_core::__private::schemars", rename = "SceneTransform")]
    pub struct StreamedTransform {
        pub translation: [f32; 3] = [0.0; 3] => range(-1.0e6, 1.0e6),
        pub rotation_degrees: [f32; 3] = [0.0; 3] => range(-360_000.0, 360_000.0),
        pub scale: [f32; 3] = [1.0; 3] => range(0.0001, 10_000.0),
    }
}

impl StreamedTransform {
    pub(crate) fn transform(self) -> Transform {
        let [x, y, z] = self.rotation_degrees.map(f32::to_radians);
        Transform {
            translation: self.translation.into(),
            rotation: Quat::from_euler(EulerRot::XYZ, x, y, z),
            scale: self.scale.into(),
        }
    }
}

#[derive(Deserialize)]
struct StreamedPrefab {
    entities: Vec<StreamedObject>,
}

fn standing() -> Option<Grounded> {
    Some(Grounded::default())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    crate = "engine_core::__private::schemars",
    rename = "SceneAlong",
    transform = along_rules
)]
pub struct StreamedAlong {
    pub id: String,
    pub prefab: String,
    pub terrain: String,
    pub layer: String,
    #[serde(default)]
    pub first: f32,
    pub every: f32,
    #[serde(default)]
    pub offset: f32,
    #[serde(default = "standing")]
    pub grounded: Option<Grounded>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streamed: Option<AlongStreamed>,
}

impl Default for StreamedAlong {
    fn default() -> Self {
        Self {
            id: String::new(),
            prefab: String::new(),
            terrain: String::new(),
            layer: String::new(),
            first: 0.0,
            every: 1.0,
            offset: 0.0,
            grounded: standing(),
            streamed: None,
        }
    }
}

fn along_rules(schema: &mut Schema) {
    for (field, rule) in [
        ("id", fields::id()),
        ("prefab", fields::id()),
        ("terrain", fields::id()),
        ("layer", fields::id()),
        ("first", fields::range(0.0, MAX_DISTANCE)),
        ("every", fields::range(MIN_EVERY, MAX_DISTANCE)),
        ("offset", fields::range(-MAX_OFFSET, MAX_OFFSET)),
    ] {
        state(schema, field, rule);
    }
}

fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    #[schemars(crate = "engine_core::__private::schemars", rename = "SceneAlongStreamed")]
    pub struct AlongStreamed {
        pub reach: f32 = 0.0 => range(0.0, MAX_REACH),
    }
}

impl StreamedAlong {
    fn validate(&self, kinds: &[ComponentKind], path: &str) -> Result<(), Error> {
        let invalid = |invalid: InvalidSetting| invalid.error(ErrorCode::InvalidValue, path);
        Setting::check(&self.id, "id", fields::id()).map_err(invalid)?;
        Setting::check(&self.prefab, "prefab", fields::id()).map_err(invalid)?;
        Setting::check(&self.terrain, "terrain", fields::id()).map_err(invalid)?;
        Setting::check(&self.layer, "layer", fields::id()).map_err(invalid)?;
        Setting::check(&self.first, "first", fields::range(0.0, MAX_DISTANCE)).map_err(invalid)?;
        Setting::check(&self.every, "every", fields::range(MIN_EVERY, MAX_DISTANCE))
            .map_err(invalid)?;
        Setting::check(
            &self.offset,
            "offset",
            fields::range(-MAX_OFFSET, MAX_OFFSET),
        )
        .map_err(invalid)?;
        if let (Some(grounded), Some(kind)) = (&self.grounded, kind(kinds, GROUNDED)) {
            kind.validate(&grounded_value(grounded), &format!("{path}.grounded"))?;
        }
        Setting::check(&self.streamed, "streamed", Rule::Any).map_err(invalid)
    }

    pub(crate) fn spots(&self, points: &[[f32; 2]], path: &str) -> Result<Vec<(Vec2, f32)>, Error> {
        let mut spots = Vec::new();
        let mut next = self.first;
        let mut walked = 0.0;
        for pair in points.windows(2) {
            let (from, to) = (Vec2::from(pair[0]), Vec2::from(pair[1]));
            let length = from.distance(to);
            if length <= 0.0 {
                continue;
            }
            let along = (to - from) / length;
            while next <= walked + length {
                if spots.len() == MAX_ALONG {
                    return Err(Error::new(ErrorCode::OutOfRange, format!("{path}.every"))
                        .at_most(MAX_ALONG as u64)
                        .got(self.every));
                }
                let point = from + along * (next - walked) + along.perp() * self.offset;
                spots.push((point, -along.y.atan2(along.x)));
                next += self.every;
            }
            walked += length;
        }
        Ok(spots)
    }
}

fn grounded_value(grounded: &Grounded) -> Value {
    serde_json::to_value(grounded).expect("grounded_serializes")
}

pub(crate) fn kind<'a>(kinds: &'a [ComponentKind], key: &str) -> Option<&'a ComponentKind> {
    kinds.iter().find(|kind| kind.key() == key)
}

pub(crate) struct ExpandedObject {
    pub(crate) source: StreamedObject,
    pub(crate) path: String,
    pub(crate) prefab_id: Option<String>,
    pub(crate) prefab_instance: Option<String>,
    pub(crate) prefab_local_id: Option<String>,
    pub(crate) reach: Option<f32>,
}

pub(crate) struct ExpandedList {
    pub(crate) entities: Vec<ExpandedObject>,
    pub(crate) transforms: Vec<Transform>,
}

pub(crate) struct Expanded {
    pub(crate) streamed: ExpandedList,
    pub(crate) kept: ExpandedList,
}

struct Listed<'a> {
    object: &'a StreamedObject,
    path: String,
    reach: Option<f32>,
}

struct Spot {
    object: StreamedObject,
    path: String,
    reach: Option<f32>,
}

fn listed<'a>(list: &'a [StreamedObject], list_path: &str) -> Vec<Listed<'a>> {
    list.iter()
        .enumerate()
        .map(|(index, object)| Listed {
            object,
            path: format!("{list_path}[{index}]"),
            reach: None,
        })
        .collect()
}

struct Prefab {
    index: usize,
    entities: Vec<StreamedObject>,
}

type Prefabs<'a> = BTreeMap<&'a str, Prefab>;

struct Names<'a> {
    meshes: BTreeSet<&'a str>,
    materials: BTreeSet<&'a str>,
    prefabs: BTreeSet<&'a str>,
}

fn prefabs(scene: &SceneContext) -> Result<Prefabs<'_>, Error> {
    scene
        .prefabs
        .iter()
        .map(|(id, (index, value))| -> Result<(&str, Prefab), Error> {
            let prefab: StreamedPrefab =
                parse_value_at(value.clone(), &format!("$.prefabs[{index}]"))?;
            Ok((
                id.as_str(),
                Prefab {
                    index: *index,
                    entities: prefab.entities,
                },
            ))
        })
        .collect()
}

impl StreamedScene {
    pub(crate) fn expand(&self, scene: &SceneContext) -> Result<Expanded, Error> {
        let prefabs = prefabs(scene)?;
        let names = Names {
            meshes: scene.assets.meshes.keys().map(String::as_str).collect(),
            materials: scene.assets.materials.keys().map(String::as_str).collect(),
            prefabs: prefabs.keys().copied().collect(),
        };
        let kinds: &[ComponentKind] = &scene.kinds;
        let placed = self.placed_along(kinds, &scene.paths)?;
        let mut kept = Vec::new();
        let mut streamed = listed(&self.streamed, "$.streamed");
        for spot in &placed {
            let entry = Listed {
                object: &spot.object,
                path: spot.path.clone(),
                reach: spot.reach,
            };
            if spot.reach.is_some() {
                streamed.push(entry);
            } else {
                kept.push(entry);
            }
        }
        let entities = scene.entities.as_ref();
        let kept = expand_list(&kept, (&names, &prefabs), kinds, Some(entities))?;
        let streamed = expand_list(&streamed, (&names, &prefabs), kinds, None)?;
        let known: BTreeSet<&str> = entities
            .keys()
            .map(String::as_str)
            .chain(kept.entities.iter().map(|entity| entity.source.id.as_str()))
            .collect();
        for entity in &streamed.entities {
            if known.contains(entity.source.id.as_str()) {
                return Err(
                    Error::new(ErrorCode::DuplicateId, format!("{}.id", entity.path))
                        .expected(Expected::Unique)
                        .got(&entity.source.id),
                );
            }
        }
        for entity in &streamed.entities {
            check_streamed(&entity.source, &entity.path, kinds)?;
        }
        Ok(Expanded { streamed, kept })
    }

    fn placed_along(
        &self,
        kinds: &[ComponentKind],
        paths: &[ScenePath],
    ) -> Result<Vec<Spot>, Error> {
        unique_ids(self.along.iter().map(|along| along.id.as_str()), "$.along")?;
        let mut placed = Vec::new();
        for (index, along) in self.along.iter().enumerate() {
            let path = format!("$.along[{index}]");
            along.validate(kinds, &path)?;
            let chosen = along_path(along, &path, paths)?;
            let origin = Vec3::from_array(chosen.origin);
            let spots = along.spots(&chosen.points, &path)?;
            for (number, (point, yaw)) in spots.into_iter().enumerate() {
                placed.push(Spot {
                    object: StreamedObject {
                        id: format!("{}{number}", along.id),
                        prefab: Some(along.prefab.clone()),
                        components: along
                            .grounded
                            .iter()
                            .map(|grounded| (GROUNDED.to_owned(), grounded_value(grounded)))
                            .collect(),
                        transform: StreamedTransform {
                            translation: (origin + Vec3::new(point.x, 0.0, point.y)).to_array(),
                            rotation_degrees: [0.0, yaw.to_degrees(), 0.0],
                            ..StreamedTransform::default()
                        },
                        ..StreamedObject::default()
                    },
                    path: path.clone(),
                    reach: along.streamed.map(|streamed| streamed.reach),
                });
            }
        }
        Ok(placed)
    }
}

fn along_path<'a>(
    along: &StreamedAlong,
    path: &str,
    paths: &'a [ScenePath],
) -> Result<&'a ScenePath, Error> {
    let found: Vec<&ScenePath> = paths
        .iter()
        .filter(|answer| answer.terrain == along.terrain)
        .collect();
    if found.is_empty() {
        let mut answered: Vec<String> = Vec::new();
        for answer in paths {
            if answered.last() != Some(&answer.terrain) {
                answered.push(answer.terrain.clone());
            }
        }
        return Err(
            Error::new(ErrorCode::UnknownReference, format!("{path}.terrain"))
                .got(&along.terrain)
                .variants(answered),
        );
    }
    let Some(chosen) = found.iter().position(|answer| answer.id == along.layer) else {
        return Err(
            Error::new(ErrorCode::UnknownReference, format!("{path}.layer"))
                .got(&along.layer)
                .variants(found.iter().map(|answer| answer.id.clone())),
        );
    };
    Ok(found[chosen])
}

fn expand_list(
    list: &[Listed<'_>],
    (names, prefabs): (&Names<'_>, &Prefabs<'_>),
    kinds: &[ComponentKind],
    taken: Option<&BTreeMap<String, Entity>>,
) -> Result<ExpandedList, Error> {
    let held = |id: &str| taken.is_some_and(|taken| taken.contains_key(id));
    let mut known = BTreeSet::new();
    for entry in list {
        let path = format!("{}.id", entry.path);
        identifier(&entry.object.id, &path)?;
        if !known.insert(entry.object.id.as_str()) || held(&entry.object.id) {
            return Err(Error::new(ErrorCode::DuplicateId, path)
                .expected(Expected::Unique)
                .got(&entry.object.id));
        }
    }
    let mut expanded_count = list.len();
    for entry in list {
        validate_object(entry.object, &entry.path, names, kinds)?;
        if let Some(id) = &entry.object.prefab {
            expanded_count += prefabs[id.as_str()].entities.len();
        }
    }
    let mut entities = Vec::with_capacity(expanded_count);
    let mut transforms = Vec::with_capacity(expanded_count);
    for entry in list {
        let entity = entry.object;
        let mut source = entity.clone();
        source.prefab = None;
        entities.push(ExpandedObject {
            source,
            path: entry.path.clone(),
            prefab_id: entity.prefab.clone(),
            prefab_instance: entity.prefab.as_ref().map(|_| entity.id.clone()),
            prefab_local_id: None,
            reach: entry.reach,
        });
        transforms.push(entity.transform.transform());
        if let Some(id) = &entity.prefab {
            let prefab = &prefabs[id.as_str()];
            for (local_index, local) in prefab.entities.iter().enumerate() {
                let path = format!("$.prefabs[{}].entities[{local_index}]", prefab.index);
                let mut source = local.clone();
                source.id = format!("{}.{}", entity.id, local.id);
                identifier(&source.id, &format!("{path}.id"))?;
                source.parent = Some(local.parent.as_ref().map_or_else(
                    || entity.id.clone(),
                    |parent| format!("{}.{}", entity.id, parent),
                ));
                entities.push(ExpandedObject {
                    source,
                    path,
                    prefab_id: Some(id.clone()),
                    prefab_instance: Some(entity.id.clone()),
                    prefab_local_id: Some(local.id.clone()),
                    reach: None,
                });
                transforms.push(local.transform.transform());
            }
        }
    }
    let mut ids = BTreeSet::new();
    for entity in &entities {
        if !ids.insert(entity.source.id.as_str()) || held(&entity.source.id) {
            return Err(
                Error::new(ErrorCode::DuplicateId, format!("{}.id", entity.path))
                    .expected(Expected::Unique)
                    .got(&entity.source.id),
            );
        }
        check_placement(&entity.source, &entity.path, kinds)?;
    }
    validate_graph(
        entities
            .iter()
            .map(|entity| (&entity.source, entity.path.clone())),
        &transforms,
        kinds,
    )?;
    Ok(ExpandedList {
        entities,
        transforms,
    })
}

fn validate_object(
    entity: &StreamedObject,
    path: &str,
    names: &Names<'_>,
    kinds: &[ComponentKind],
) -> Result<Transform, Error> {
    for (name, value) in entity.authored() {
        if let Some(kind) = kind(kinds, name) {
            kind.validate(value, &format!("{path}.{name}"))?;
        }
    }
    match (&entity.mesh, &entity.material) {
        (Some(mesh), Some(material)) if entity.prefab.is_none() => {
            reference(mesh, &names.meshes, &format!("{path}.mesh"))?;
            reference(material, &names.materials, &format!("{path}.material"))?;
        }
        (None, None) => {}
        (Some(_), Some(_)) => {
            return Err(
                Error::new(ErrorCode::InvalidRenderable, format!("{path}.mesh"))
                    .expected(Expected::Absent),
            );
        }
        (Some(_), None) => {
            return Err(
                Error::new(ErrorCode::InvalidRenderable, format!("{path}.material"))
                    .expected(Expected::Present),
            );
        }
        (None, Some(_)) => {
            return Err(
                Error::new(ErrorCode::InvalidRenderable, format!("{path}.mesh"))
                    .expected(Expected::Present),
            );
        }
    }
    if let Some(prefab) = &entity.prefab {
        reference(prefab, &names.prefabs, &format!("{path}.prefab"))?;
    }
    entity
        .validate()
        .map_err(|invalid| invalid.error(ErrorCode::InvalidValue, path))?;
    if let Some(parent) = &entity.parent {
        identifier(parent, &format!("{path}.parent"))?;
    }
    display_name(&entity.name, &format!("{path}.name"))?;
    let mut tags = BTreeSet::new();
    for (index, tag) in entity.tags.iter().enumerate() {
        let path = format!("{path}.tags[{index}]");
        identifier(tag, &path)?;
        if !tags.insert(tag) {
            return Err(Error::new(ErrorCode::DuplicateTag, path)
                .expected(Expected::Unique)
                .got(tag));
        }
    }
    Ok(entity.transform.transform())
}

fn check_placement(
    entity: &StreamedObject,
    path: &str,
    kinds: &[ComponentKind],
) -> Result<(), Error> {
    if entity.parent.is_none() {
        return Ok(());
    }
    refused_placement(entity, path, kinds, |placement| match placement {
        Placement::Roots(code) => Some(code),
        Placement::Anywhere | Placement::Entities(_) => None,
    })
}

fn check_streamed(
    entity: &StreamedObject,
    path: &str,
    kinds: &[ComponentKind],
) -> Result<(), Error> {
    refused_placement(entity, path, kinds, |placement| match placement {
        Placement::Entities(code) => Some(code),
        Placement::Anywhere | Placement::Roots(_) => None,
    })
}

fn refused_placement(
    entity: &StreamedObject,
    path: &str,
    kinds: &[ComponentKind],
    refused: fn(Placement) -> Option<ErrorCode>,
) -> Result<(), Error> {
    for (name, _) in entity.authored() {
        if let Some(code) = kind(kinds, name).and_then(|kind| refused(kind.placement())) {
            return Err(Error::new(code, format!("{path}.{name}")).expected(Expected::Absent));
        }
    }
    Ok(())
}

fn validate_graph<'a>(
    entries: impl Iterator<Item = (&'a StreamedObject, String)>,
    transforms: &[Transform],
    kinds: &[ComponentKind],
) -> Result<(), Error> {
    let entries: Vec<_> = entries.collect();
    let ids: BTreeMap<_, _> = entries
        .iter()
        .enumerate()
        .map(|(index, (entity, _))| (entity.id.as_str(), index))
        .collect();
    let mut parents = Vec::with_capacity(entries.len());
    for (entity, path) in &entries {
        parents.push(
            entity
                .parent
                .as_ref()
                .map(|parent| {
                    ids.get(parent.as_str()).copied().ok_or_else(|| {
                        unknown(
                            Error::new(ErrorCode::UnknownParent, format!("{path}.parent")),
                            parent,
                            ids.keys().copied(),
                        )
                    })
                })
                .transpose()?,
        );
    }
    let mut worlds: Vec<Option<Mat4>> = vec![None; entries.len()];
    let mut depths = vec![0usize; entries.len()];
    let mut path = Vec::with_capacity(MAX_DEPTH + 1);
    for start in 0..entries.len() {
        if worlds[start].is_some() {
            continue;
        }
        path.clear();
        let mut cursor = Some(start);
        while let Some(index) = cursor {
            if worlds[index].is_some() {
                break;
            }
            if path.contains(&index) {
                return Err(Error::new(
                    ErrorCode::HierarchyCycle,
                    format!("{}.parent", entries[index].1),
                )
                .expected(Expected::Acyclic)
                .got(&entries[index].0.parent));
            }
            if path.len() > MAX_DEPTH {
                return Err(Error::new(
                    ErrorCode::HierarchyDepth,
                    format!("{}.parent", entries[start].1),
                )
                .at_most(MAX_DEPTH as u64)
                .got(path.len()));
            }
            path.push(index);
            cursor = parents[index];
        }
        while let Some(index) = path.pop() {
            let (parent_matrix, depth) = parents[index].map_or((Mat4::IDENTITY, 0), |parent| {
                (
                    worlds[parent].expect("parent_resolved_first"),
                    depths[parent] + 1,
                )
            });
            if depth > MAX_DEPTH {
                return Err(Error::new(
                    ErrorCode::HierarchyDepth,
                    format!("{}.parent", entries[index].1),
                )
                .at_most(MAX_DEPTH as u64)
                .got(depth));
            }
            let local = &transforms[index];
            let matrix = parent_matrix
                * Mat4::from_scale_rotation_translation(
                    local.scale,
                    local.rotation,
                    local.translation,
                );
            if !valid_global_matrix(matrix) {
                return Err(Error::new(
                    ErrorCode::InvalidWorldTransform,
                    format!("{}.transform", entries[index].1),
                )
                .expected(Expected::Invertible)
                .got(entries[index].0.transform));
            }
            let (entity, at) = &entries[index];
            let placed = Placed { world: matrix };
            for (name, value) in entity.authored() {
                if let Some(kind) = kind(kinds, name) {
                    kind.validate_placed(value, placed, &format!("{at}.{name}"))?;
                }
            }
            worlds[index] = Some(matrix);
            depths[index] = depth;
        }
    }
    Ok(())
}

fn identifier(id: &str, path: &str) -> Result<(), Error> {
    if is_id(id) {
        return Ok(());
    }
    Err(Error::new(ErrorCode::InvalidId, path)
        .expected(Expected::Id)
        .at_most(MAX_ID_BYTES as u64)
        .got(id))
}

fn display_name(name: &str, path: &str) -> Result<(), Error> {
    if name.len() > MAX_NAME_BYTES || name.chars().any(char::is_control) {
        return Err(Error::new(ErrorCode::InvalidName, path)
            .at_most(MAX_NAME_BYTES as u64)
            .got(name));
    }
    Ok(())
}

fn unique_ids<'a>(
    ids: impl Iterator<Item = &'a str>,
    collection_path: &str,
) -> Result<BTreeSet<&'a str>, Error> {
    let mut known = BTreeSet::new();
    for (index, id) in ids.enumerate() {
        let path = format!("{collection_path}[{index}].id");
        identifier(id, &path)?;
        if !known.insert(id) {
            return Err(Error::new(ErrorCode::DuplicateId, path)
                .expected(Expected::Unique)
                .got(id));
        }
    }
    Ok(known)
}

fn reference(id: &str, known: &BTreeSet<&str>, path: &str) -> Result<(), Error> {
    identifier(id, path)?;
    if !known.contains(id) {
        return Err(unknown(
            Error::new(ErrorCode::UnknownReference, path),
            id,
            known.iter().copied(),
        ));
    }
    Ok(())
}

fn unknown<S: Into<String>>(error: Error, id: &str, known: impl IntoIterator<Item = S>) -> Error {
    let known: Vec<String> = known.into_iter().map(Into::into).collect();
    let error = error.got(id);
    match closest(id, &known) {
        Some(nearest) => error.nearest(nearest),
        None => error,
    }
}
