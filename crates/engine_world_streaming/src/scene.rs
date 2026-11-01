use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::prelude::Resource;
use engine_codec::{Expected, closest};
use engine_core::__private::schemars::JsonSchema;
use engine_core::{
    Error, ErrorCode, MAX_HIERARCHY_DEPTH as MAX_DEPTH, MaterialHandle, MeshHandle, Section,
    Transform, is_id, valid_global_matrix,
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

const MAX_PREFAB_ENTITIES: usize = 10_000;

const MAX_TRANSLATION: f32 = 1.0e6;

const MAX_ROTATION_DEGREES: f32 = 360_000.0;

const SCALE: (f32, f32) = (0.0001, 10_000.0);

#[derive(Clone, Debug, Default, PartialEq, Resource, Serialize, Deserialize, JsonSchema)]
#[component(immutable)]
#[serde(default, deny_unknown_fields)]
#[schemars(crate = "engine_core::__private::schemars", rename = "SceneStreamed")]
pub struct StreamedScene {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub objects: Vec<StreamedObject>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub prefabs: Vec<StreamedPrefab>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub along: Vec<StreamedAlong>,
}

impl Section for StreamedScene {
    const KEY: &'static str = "streamed";

    fn check(&self) -> Result<(), Error> {
        for (index, along) in self.along.iter().enumerate() {
            along.check(&format!("streamed.along[{index}]"))?;
        }
        self.expand(&[]).map(|_| ())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(crate = "engine_core::__private::schemars")]
pub struct StreamedObject {
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<String>")]
    pub mesh: Option<MeshHandle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<String>")]
    pub material: Option<MaterialHandle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefab: Option<String>,
    #[serde(flatten)]
    #[schemars(skip)]
    pub components: BTreeMap<String, Value>,
    #[serde(default)]
    pub transform: StreamedTransform,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
#[schemars(crate = "engine_core::__private::schemars")]
pub struct StreamedTransform {
    pub translation: [f32; 3],
    pub rotation_degrees: [f32; 3],
    pub scale: [f32; 3],
}

impl Default for StreamedTransform {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation_degrees: [0.0; 3],
            scale: [1.0; 3],
        }
    }
}

impl StreamedTransform {
    fn transform(self) -> Transform {
        let [x, y, z] = self.rotation_degrees.map(f32::to_radians);
        Transform {
            translation: self.translation.into(),
            rotation: Quat::from_euler(EulerRot::XYZ, x, y, z),
            scale: self.scale.into(),
        }
    }

    fn check(&self, path: &str) -> Result<(), Error> {
        let fields = [
            (
                "translation",
                self.translation,
                (-MAX_TRANSLATION, MAX_TRANSLATION),
            ),
            (
                "rotation_degrees",
                self.rotation_degrees,
                (-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES),
            ),
            ("scale", self.scale, SCALE),
        ];
        for (field, values, (low, high)) in fields {
            if !values
                .iter()
                .all(|value| value.is_finite() && (low..=high).contains(value))
            {
                return Err(
                    Error::new(ErrorCode::InvalidValue, format!("{path}.transform.{field}"))
                        .range(f64::from(low), f64::from(high))
                        .got(values),
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(crate = "engine_core::__private::schemars")]
pub struct StreamedPrefab {
    pub id: String,
    pub entities: Vec<StreamedObject>,
}

fn standing() -> Option<Grounded> {
    Some(Grounded::default())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(crate = "engine_core::__private::schemars")]
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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(crate = "engine_core::__private::schemars")]
pub struct AlongStreamed {
    pub reach: f32,
}

impl StreamedAlong {
    fn check(&self, path: &str) -> Result<(), Error> {
        for (field, id) in [
            ("prefab", &self.prefab),
            ("terrain", &self.terrain),
            ("layer", &self.layer),
        ] {
            identifier(id, &format!("{path}.{field}"))?;
        }
        let reach = self.streamed.map_or(0.0, |streamed| streamed.reach);
        for (field, value, low, high) in [
            ("first", self.first, 0.0, MAX_DISTANCE),
            ("every", self.every, MIN_EVERY, MAX_DISTANCE),
            ("offset", self.offset, -MAX_OFFSET, MAX_OFFSET),
            ("streamed.reach", reach, 0.0, MAX_REACH),
        ] {
            if !value.is_finite() || !(low..=high).contains(&value) {
                return Err(Error::new(ErrorCode::InvalidValue, format!("{path}.{field}"))
                    .range(f64::from(low), f64::from(high))
                    .got(value));
            }
        }
        Ok(())
    }

    pub(crate) fn spots(
        &self,
        points: &[[f32; 2]],
        path: &str,
    ) -> Result<Vec<(Vec2, f32)>, Error> {
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

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Placed {
    pub(crate) along: usize,
    pub(crate) origin: Vec3,
    pub(crate) spots: Vec<(Vec2, f32)>,
}

pub(crate) struct ExpandedObject {
    pub(crate) source: StreamedObject,
    pub(crate) path: String,
    pub(crate) grounded: Option<Grounded>,
    pub(crate) reach: Option<f32>,
    pub(crate) transform: Transform,
}

pub(crate) struct Expanded {
    pub(crate) streamed: Vec<ExpandedObject>,
    pub(crate) kept: Vec<ExpandedObject>,
}

struct Listed<'a> {
    object: &'a StreamedObject,
    path: String,
    grounded: Option<Grounded>,
    reach: Option<f32>,
}

type Prefabs<'a> = BTreeMap<&'a str, (usize, &'a StreamedPrefab)>;

impl StreamedScene {
    pub(crate) fn expand(&self, placed: &[Placed]) -> Result<Expanded, Error> {
        let (prefab_ids, prefabs) = self.validate_prefabs()?;
        unique_ids(self.along.iter().map(|along| along.id.as_str()), "streamed.along")?;
        for (index, along) in self.along.iter().enumerate() {
            reference(
                &along.prefab,
                &prefab_ids,
                &format!("streamed.along[{index}].prefab"),
            )?;
        }
        let spots = self.placed_along(placed);
        let mut streamed: Vec<Listed<'_>> = self
            .objects
            .iter()
            .enumerate()
            .map(|(index, object)| Listed {
                object,
                path: format!("streamed.objects[{index}]"),
                grounded: None,
                reach: None,
            })
            .collect();
        let mut kept = Vec::new();
        for (object, path, grounded, reach) in &spots {
            let entry = Listed {
                object,
                path: path.clone(),
                grounded: *grounded,
                reach: *reach,
            };
            if reach.is_some() {
                streamed.push(entry);
            } else {
                kept.push(entry);
            }
        }
        let streamed = expand_list(&streamed, &prefab_ids, &prefabs)?;
        let kept = expand_list(&kept, &prefab_ids, &prefabs)?;
        let known: BTreeSet<&str> = kept
            .iter()
            .map(|entity| entity.source.id.as_str())
            .collect();
        for entity in &streamed {
            if known.contains(entity.source.id.as_str()) {
                return Err(
                    Error::new(ErrorCode::DuplicateId, format!("{}.id", entity.path))
                        .expected(Expected::Unique)
                        .got(&entity.source.id),
                );
            }
        }
        Ok(Expanded { streamed, kept })
    }

    fn validate_prefabs(&self) -> Result<(BTreeSet<&str>, Prefabs<'_>), Error> {
        let prefab_ids = unique_ids(
            self.prefabs.iter().map(|prefab| prefab.id.as_str()),
            "streamed.prefabs",
        )?;
        for (index, prefab) in self.prefabs.iter().enumerate() {
            let path = format!("streamed.prefabs[{index}].entities");
            if prefab.entities.len() > MAX_PREFAB_ENTITIES {
                return Err(Error::new(ErrorCode::InvalidValue, path)
                    .at_most(MAX_PREFAB_ENTITIES as u64)
                    .got(prefab.entities.len()));
            }
            unique_ids(
                prefab.entities.iter().map(|entity| entity.id.as_str()),
                &path,
            )?;
            let mut transforms = Vec::with_capacity(prefab.entities.len());
            for (local_index, entity) in prefab.entities.iter().enumerate() {
                let path = format!("{path}[{local_index}]");
                if let Some(nested) = &entity.prefab {
                    return Err(
                        Error::new(ErrorCode::NestedPrefab, format!("{path}.prefab"))
                            .expected(Expected::Absent)
                            .got(nested),
                    );
                }
                transforms.push(validate_object(entity, &path, &prefab_ids)?);
            }
            validate_graph(
                prefab
                    .entities
                    .iter()
                    .enumerate()
                    .map(|(index, entity)| (entity, format!("{path}[{index}]"))),
                &transforms,
            )?;
        }
        let prefab_lookup: Prefabs<'_> = self
            .prefabs
            .iter()
            .enumerate()
            .map(|(index, prefab)| (prefab.id.as_str(), (index, prefab)))
            .collect();
        Ok((prefab_ids, prefab_lookup))
    }

    fn placed_along(
        &self,
        placed: &[Placed],
    ) -> Vec<(StreamedObject, String, Option<Grounded>, Option<f32>)> {
        let mut objects = Vec::new();
        for spot in placed {
            let along = &self.along[spot.along];
            let path = format!("streamed.along[{}]", spot.along);
            for (number, &(point, yaw)) in spot.spots.iter().enumerate() {
                objects.push((
                    StreamedObject {
                        id: format!("{}{number}", along.id),
                        prefab: Some(along.prefab.clone()),
                        transform: StreamedTransform {
                            translation: (spot.origin + Vec3::new(point.x, 0.0, point.y))
                                .to_array(),
                            rotation_degrees: [0.0, yaw.to_degrees(), 0.0],
                            ..StreamedTransform::default()
                        },
                        ..StreamedObject::default()
                    },
                    path.clone(),
                    along.grounded,
                    along.streamed.map(|streamed| streamed.reach),
                ));
            }
        }
        objects
    }
}

fn expand_list(
    list: &[Listed<'_>],
    prefab_ids: &BTreeSet<&str>,
    prefab_lookup: &Prefabs<'_>,
) -> Result<Vec<ExpandedObject>, Error> {
    let mut known = BTreeSet::new();
    for entry in list {
        let path = format!("{}.id", entry.path);
        identifier(&entry.object.id, &path)?;
        if !known.insert(entry.object.id.as_str()) {
            return Err(Error::new(ErrorCode::DuplicateId, path)
                .expected(Expected::Unique)
                .got(&entry.object.id));
        }
    }
    let mut expanded_count = list.len();
    for entry in list {
        validate_object(entry.object, &entry.path, prefab_ids)?;
        if let Some(id) = &entry.object.prefab {
            expanded_count += prefab_lookup[id.as_str()].1.entities.len();
        }
    }
    let mut entities = Vec::with_capacity(expanded_count);
    for entry in list {
        let entity = entry.object;
        let mut source = entity.clone();
        source.prefab = None;
        entities.push(ExpandedObject {
            source,
            path: entry.path.clone(),
            grounded: entry.grounded,
            reach: entry.reach,
            transform: entity.transform.transform(),
        });
        if let Some(id) = &entity.prefab {
            let (prefab_index, prefab) = prefab_lookup[id.as_str()];
            for (local_index, local) in prefab.entities.iter().enumerate() {
                let path = format!("streamed.prefabs[{prefab_index}].entities[{local_index}]");
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
                    grounded: None,
                    reach: None,
                    transform: local.transform.transform(),
                });
            }
        }
    }
    let mut ids = BTreeSet::new();
    for entity in &entities {
        if !ids.insert(entity.source.id.as_str()) {
            return Err(
                Error::new(ErrorCode::DuplicateId, format!("{}.id", entity.path))
                    .expected(Expected::Unique)
                    .got(&entity.source.id),
            );
        }
    }
    let transforms: Vec<Transform> = entities.iter().map(|entity| entity.transform).collect();
    validate_graph(
        entities
            .iter()
            .map(|entity| (&entity.source, entity.path.clone())),
        &transforms,
    )?;
    Ok(entities)
}

fn validate_object(
    entity: &StreamedObject,
    path: &str,
    prefab_ids: &BTreeSet<&str>,
) -> Result<Transform, Error> {
    match (&entity.mesh, &entity.material) {
        (Some(_), Some(_)) if entity.prefab.is_none() => {}
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
        reference(prefab, prefab_ids, &format!("{path}.prefab"))?;
    }
    entity.transform.check(path)?;
    if let Some(parent) = &entity.parent {
        identifier(parent, &format!("{path}.parent"))?;
    }
    Ok(entity.transform.transform())
}

fn validate_graph<'a>(
    entries: impl Iterator<Item = (&'a StreamedObject, String)>,
    transforms: &[Transform],
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
        .got(id))
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

fn unknown<S: Into<String>>(
    error: Error,
    id: &str,
    known: impl IntoIterator<Item = S>,
) -> Error {
    let known: Vec<String> = known.into_iter().map(Into::into).collect();
    let error = error.got(id);
    match closest(id, &known) {
        Some(nearest) => error.nearest(nearest),
        None => error,
    }
}
