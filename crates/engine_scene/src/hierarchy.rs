use engine_settings::ErrorCode;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

use engine_core::{
    Error, Expected, MAX_HIERARCHY_DEPTH as MAX_DEPTH, Transform,
    glam::{Mat4, Vec3},
    valid_global_matrix,
};

use crate::components::GROUNDED;
use crate::rules::PATH_LAYER;
use crate::terrain::TERRAIN;
use crate::validation::{display_name, identifier, reference, unique_ids, unknown};
use crate::{SceneAlong, SceneDocument, SceneObject, ScenePrefab, SceneTerrain, SceneTransform};

pub(crate) struct ExpandedObject {
    pub source: SceneObject,
    pub path: String,
    pub prefab_id: Option<String>,
    pub prefab_instance: Option<String>,
    pub prefab_local_id: Option<String>,
    pub reach: Option<f32>,
}

struct Listed<'a> {
    object: &'a SceneObject,
    path: String,
    reach: Option<f32>,
}

struct Placed {
    object: SceneObject,
    path: String,
    reach: Option<f32>,
}

fn listed<'a>(list: &'a [SceneObject], list_path: &str) -> Vec<Listed<'a>> {
    list.iter()
        .enumerate()
        .map(|(index, object)| Listed {
            object,
            path: format!("{list_path}[{index}]"),
            reach: None,
        })
        .collect()
}

pub(crate) struct ExpandedScene {
    pub entities: Vec<ExpandedObject>,
    pub transforms: Vec<Transform>,
}

type Prefabs<'a> = BTreeMap<&'a str, (usize, &'a ScenePrefab)>;

impl SceneDocument {
    pub fn expanded_entities(&self) -> Result<Vec<SceneObject>, Error> {
        Ok(self
            .prepare(None)?
            .entities
            .into_iter()
            .map(|entity| entity.source)
            .collect())
    }

    pub(crate) fn expand(
        &self,
        mesh_ids: &BTreeSet<&str>,
        material_ids: &BTreeSet<&str>,
    ) -> Result<[ExpandedScene; 2], Error> {
        let (prefab_ids, prefabs) = self.validate_prefabs(mesh_ids, material_ids)?;
        let ids = (mesh_ids, material_ids, &prefab_ids);
        let placed = self.placed_along()?;
        let mut kept = listed(&self.entities, "$.entities");
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
        let entities = expand_list(&kept, ids, &prefabs)?;
        let streamed = expand_list(&streamed, ids, &prefabs)?;
        let known: BTreeSet<&str> = entities
            .entities
            .iter()
            .map(|entity| entity.source.id.as_str())
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
        Ok([entities, streamed])
    }

    fn validate_prefabs(
        &self,
        mesh_ids: &BTreeSet<&str>,
        material_ids: &BTreeSet<&str>,
    ) -> Result<(BTreeSet<&str>, Prefabs<'_>), Error> {
        let prefab_ids = unique_ids(
            self.prefabs.iter().map(|prefab| prefab.id.as_str()),
            "$.prefabs",
        )?;
        for (index, prefab) in self.prefabs.iter().enumerate() {
            prefab.validate().map_err(|invalid| {
                invalid.error(ErrorCode::InvalidValue, &format!("$.prefabs[{index}]"))
            })?;
            let path = format!("$.prefabs[{index}].entities");
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
                transforms.push(validate_object(
                    entity,
                    &path,
                    mesh_ids,
                    material_ids,
                    &prefab_ids,
                )?);
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

    fn placed_along(&self) -> Result<Vec<Placed>, Error> {
        unique_ids(self.along.iter().map(|along| along.id.as_str()), "$.along")?;
        let mut placed = Vec::new();
        for (index, along) in self.along.iter().enumerate() {
            let path = format!("$.along[{index}]");
            along
                .validate()
                .map_err(|invalid| invalid.error(ErrorCode::InvalidValue, &path))?;
            let (origin, points) = self.along_path(along, &path)?;
            let origin = Vec3::from_array(origin);
            for (number, (point, yaw)) in along.spots(&points, &path)?.into_iter().enumerate() {
                placed.push(Placed {
                    object: SceneObject {
                        id: format!("{}{number}", along.id),
                        prefab: Some(along.prefab.clone()),
                        components: along
                            .grounded
                            .iter()
                            .map(|grounded| {
                                (
                                    GROUNDED.name.to_owned(),
                                    serde_json::to_value(grounded).expect("grounded_serializes"),
                                )
                            })
                            .collect(),
                        transform: SceneTransform {
                            translation: (origin + Vec3::new(point.x, 0.0, point.y)).to_array(),
                            rotation_degrees: [0.0, yaw.to_degrees(), 0.0],
                            ..SceneTransform::default()
                        },
                        ..SceneObject::default()
                    },
                    path: path.clone(),
                    reach: along.streamed.map(|streamed| streamed.reach),
                });
            }
        }
        Ok(placed)
    }

    fn along_path(
        &self,
        along: &SceneAlong,
        path: &str,
    ) -> Result<([f32; 3], Vec<[f32; 2]>), Error> {
        let Some((at, terrain)) = self
            .entities
            .iter()
            .enumerate()
            .find_map(|(index, entity)| {
                entity
                    .component(TERRAIN.name)
                    .filter(|_| entity.id == along.terrain)
                    .map(|terrain| (index, terrain))
            })
        else {
            return Err(
                Error::new(ErrorCode::UnknownReference, format!("{path}.terrain"))
                    .got(&along.terrain)
                    .variants(
                        self.entities
                            .iter()
                            .filter(|entity| entity.component(TERRAIN.name).is_some())
                            .map(|entity| entity.id.clone()),
                    ),
            );
        };
        let terrain: SceneTerrain = engine_core::parse_value_at(
            terrain.clone(),
            &format!("$.entities[{at}].{}", TERRAIN.name),
        )?;
        let paths = || {
            terrain
                .layers
                .iter()
                .enumerate()
                .filter(|(_, layer)| layer.get("type").and_then(Value::as_str) == Some(PATH_LAYER))
                .filter_map(|(index, layer)| {
                    layer
                        .get("id")
                        .and_then(Value::as_str)
                        .map(|id| (index, id, layer))
                })
        };
        let Some((index, _, layer)) = paths().find(|&(_, id, _)| id == along.layer) else {
            return Err(
                Error::new(ErrorCode::UnknownReference, format!("{path}.layer"))
                    .got(&along.layer)
                    .variants(paths().map(|(_, id, _)| id.to_owned())),
            );
        };
        let points = engine_core::parse_value_at(
            layer.get("points").cloned().unwrap_or(Value::Null),
            &format!("$.entities[{at}].terrain.layers[{index}].points"),
        )?;
        Ok((terrain.origin, points))
    }
}

fn expand_list(
    list: &[Listed<'_>],
    (mesh_ids, material_ids, prefab_ids): (&BTreeSet<&str>, &BTreeSet<&str>, &BTreeSet<&str>),
    prefab_lookup: &Prefabs<'_>,
) -> Result<ExpandedScene, Error> {
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
        validate_object(
            entry.object,
            &entry.path,
            mesh_ids,
            material_ids,
            prefab_ids,
        )?;
        if let Some(id) = &entry.object.prefab {
            expanded_count += prefab_lookup[id.as_str()].1.entities.len();
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
            let (prefab_index, prefab) = prefab_lookup[id.as_str()];
            for (local_index, local) in prefab.entities.iter().enumerate() {
                let path = format!("$.prefabs[{prefab_index}].entities[{local_index}]");
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
        if !ids.insert(entity.source.id.as_str()) {
            return Err(
                Error::new(ErrorCode::DuplicateId, format!("{}.id", entity.path))
                    .expected(Expected::Unique)
                    .got(&entity.source.id),
            );
        }
        entity.source.check_placement(&entity.path)?;
    }
    validate_graph(
        entities
            .iter()
            .map(|entity| (&entity.source, entity.path.clone())),
        &transforms,
    )?;
    Ok(ExpandedScene {
        entities,
        transforms,
    })
}

fn validate_object(
    entity: &SceneObject,
    path: &str,
    mesh_ids: &BTreeSet<&str>,
    material_ids: &BTreeSet<&str>,
    prefab_ids: &BTreeSet<&str>,
) -> Result<Transform, Error> {
    entity.validate_components(path)?;
    match (&entity.mesh, &entity.material) {
        (Some(mesh), Some(material)) if entity.prefab.is_none() => {
            reference(mesh, mesh_ids, &format!("{path}.mesh"))?;
            reference(material, material_ids, &format!("{path}.material"))?;
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
        reference(prefab, prefab_ids, &format!("{path}.prefab"))?;
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

fn validate_graph<'a>(
    entries: impl Iterator<Item = (&'a SceneObject, String)>,
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
            entries[index]
                .0
                .validate_world_components(matrix, &entries[index].1)?;
            worlds[index] = Some(matrix);
            depths[index] = depth;
        }
    }
    Ok(())
}
