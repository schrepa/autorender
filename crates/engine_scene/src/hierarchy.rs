use engine_settings::{ErrorCode, Expected};
use std::collections::{BTreeMap, BTreeSet};

use engine_core::{
    Error, MAX_HIERARCHY_DEPTH as MAX_DEPTH, Placed, Transform, glam::Mat4, valid_global_matrix,
};

use crate::validation::{display_name, identifier, reference, unique_ids, unknown};
use crate::{SceneDocument, SceneKinds, SceneObject, ScenePrefab};

pub(crate) struct ExpandedObject {
    pub source: SceneObject,
    pub path: String,
    pub prefab_id: Option<String>,
    pub prefab_instance: Option<String>,
    pub prefab_local_id: Option<String>,
}

struct Listed<'a> {
    object: &'a SceneObject,
    path: String,
}

fn listed<'a>(list: &'a [SceneObject], list_path: &str) -> Vec<Listed<'a>> {
    list.iter()
        .enumerate()
        .map(|(index, object)| Listed {
            object,
            path: format!("{list_path}[{index}]"),
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
            .prepare(None, &SceneKinds::default())?
            .entities
            .into_iter()
            .map(|entity| entity.source)
            .collect())
    }

    pub(crate) fn expand(
        &self,
        mesh_ids: &BTreeSet<&str>,
        material_ids: &BTreeSet<&str>,
        kinds: &SceneKinds,
    ) -> Result<ExpandedScene, Error> {
        let (prefab_ids, prefabs) = self.validate_prefabs(mesh_ids, material_ids, kinds)?;
        let ids = (mesh_ids, material_ids, &prefab_ids);
        expand_list(&listed(&self.entities, "$.entities"), ids, &prefabs, kinds)
    }

    fn validate_prefabs(
        &self,
        mesh_ids: &BTreeSet<&str>,
        material_ids: &BTreeSet<&str>,
        kinds: &SceneKinds,
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
                    (mesh_ids, material_ids, &prefab_ids),
                    kinds,
                )?);
            }
            validate_graph(
                prefab
                    .entities
                    .iter()
                    .enumerate()
                    .map(|(index, entity)| (entity, format!("{path}[{index}]"))),
                &transforms,
                kinds,
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
}

type Ids<'a, 'b> = (&'b BTreeSet<&'a str>, &'b BTreeSet<&'a str>, &'b BTreeSet<&'a str>);

fn expand_list(
    list: &[Listed<'_>],
    ids: Ids<'_, '_>,
    prefab_lookup: &Prefabs<'_>,
    kinds: &SceneKinds,
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
        validate_object(entry.object, &entry.path, ids, kinds)?;
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
        kinds.check_placement(&entity.source, &entity.path)?;
    }
    validate_graph(
        entities
            .iter()
            .map(|entity| (&entity.source, entity.path.clone())),
        &transforms,
        kinds,
    )?;
    Ok(ExpandedScene {
        entities,
        transforms,
    })
}

fn validate_object(
    entity: &SceneObject,
    path: &str,
    (mesh_ids, material_ids, prefab_ids): Ids<'_, '_>,
    kinds: &SceneKinds,
) -> Result<Transform, Error> {
    kinds.check(entity, path)?;
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
    kinds: &SceneKinds,
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
            kinds.check_placed(
                entries[index].0,
                Placed { world: matrix },
                &entries[index].1,
            )?;
            worlds[index] = Some(matrix);
            depths[index] = depth;
        }
    }
    Ok(())
}
