use std::collections::{BTreeMap, BTreeSet};

use engine_core::{MAX_HIERARCHY_DEPTH as MAX_DEPTH, Transform, glam::Mat4, valid_global_matrix};

use crate::validation::{identifier, reference, transform, unique_ids};
use crate::{SceneDocument, SceneError, SceneObject};

pub(crate) struct ExpandedObject {
    pub source: SceneObject,
    pub path: String,
    pub prefab_id: Option<String>,
    pub prefab_instance: Option<String>,
    pub prefab_local_id: Option<String>,
}

pub(crate) struct ExpandedScene {
    pub entities: Vec<ExpandedObject>,
    pub transforms: Vec<Transform>,
}

impl SceneDocument {
    /// Returns validated entities with prefab definitions expanded once.
    ///
    /// Instance roots retain their authored ID. Template entity IDs become
    /// `instance.local`; local roots attach to the instance root and local
    /// parents become qualified IDs. Prefab references are cleared in this flat
    /// result. Runtime setup should normally use [`crate::SceneInstance::entities`]
    /// and [`crate::SceneEntity`] metadata instead of expanding a second time.
    pub fn expanded_entities(&self) -> Result<Vec<SceneObject>, SceneError> {
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
    ) -> Result<ExpandedScene, SceneError> {
        unique_ids(
            self.entities.iter().map(|entity| entity.id.as_str()),
            "$.entities",
        )?;
        let prefab_ids = unique_ids(
            self.prefabs.iter().map(|prefab| prefab.id.as_str()),
            "$.prefabs",
        )?;
        for (index, prefab) in self.prefabs.iter().enumerate() {
            let path = format!("$.prefabs[{index}].entities");
            unique_ids(
                prefab.entities.iter().map(|entity| entity.id.as_str()),
                &path,
            )?;
            let mut transforms = Vec::with_capacity(prefab.entities.len());
            for (local_index, entity) in prefab.entities.iter().enumerate() {
                let path = format!("{path}[{local_index}]");
                if entity.prefab.is_some() {
                    return Err(SceneError::new(
                        "nested_prefab",
                        format!("{path}.prefab"),
                        "prefab definitions cannot reference another prefab",
                    ));
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

        let prefab_lookup: BTreeMap<_, _> = self
            .prefabs
            .iter()
            .enumerate()
            .map(|(index, prefab)| (prefab.id.as_str(), (index, prefab)))
            .collect();
        let mut expanded_count = self.entities.len();
        for (index, entity) in self.entities.iter().enumerate() {
            let path = format!("$.entities[{index}]");
            validate_object(entity, &path, mesh_ids, material_ids, &prefab_ids)?;
            if let Some(id) = &entity.prefab {
                expanded_count += prefab_lookup[id.as_str()].1.entities.len();
            }
        }
        let mut entities = Vec::with_capacity(expanded_count);
        let mut transforms = Vec::with_capacity(expanded_count);
        for (index, entity) in self.entities.iter().enumerate() {
            let mut source = entity.clone();
            source.prefab = None;
            entities.push(ExpandedObject {
                source,
                path: format!("$.entities[{index}]"),
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
                return Err(SceneError::new(
                    "duplicate_id",
                    format!("{}.id", entity.path),
                    format!("expanded entity ID '{}' is already used", entity.source.id),
                ));
            }
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
}

fn validate_object(
    entity: &SceneObject,
    path: &str,
    mesh_ids: &BTreeSet<&str>,
    material_ids: &BTreeSet<&str>,
    prefab_ids: &BTreeSet<&str>,
) -> Result<Transform, SceneError> {
    if let Some(collider) = entity.collider {
        collider.runtime(&format!("{path}.collider"))?;
    }
    match (&entity.mesh, &entity.material) {
        (Some(mesh), Some(material)) if entity.prefab.is_none() => {
            reference(mesh, mesh_ids, &format!("{path}.mesh"))?;
            reference(material, material_ids, &format!("{path}.material"))?;
        }
        (None, None) => {}
        _ => {
            return Err(SceneError::new(
                "invalid_renderable",
                path,
                "an entity requires both mesh and material, or neither; prefab instance roots cannot also render",
            ));
        }
    }
    if let Some(prefab) = &entity.prefab {
        reference(prefab, prefab_ids, &format!("{path}.prefab"))?;
    }
    if let Some(parent) = &entity.parent {
        identifier(parent, &format!("{path}.parent"))?;
    }
    if entity.name.len() > 128 || entity.name.chars().any(char::is_control) {
        return Err(SceneError::new(
            "invalid_name",
            format!("{path}.name"),
            "names must be at most 128 UTF-8 bytes without control characters",
        ));
    }
    let mut tags = BTreeSet::new();
    for (index, tag) in entity.tags.iter().enumerate() {
        let path = format!("{path}.tags[{index}]");
        identifier(tag, &path)?;
        if !tags.insert(tag) {
            return Err(SceneError::new(
                "duplicate_tag",
                path,
                format!("tag '{tag}' is repeated"),
            ));
        }
    }
    transform(&entity.transform, &format!("{path}.transform"))
}

fn validate_graph<'a>(
    entries: impl Iterator<Item = (&'a SceneObject, String)>,
    transforms: &[Transform],
) -> Result<(), SceneError> {
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
                        SceneError::new(
                            "unknown_parent",
                            format!("{path}.parent"),
                            format!("no entity in this scope has ID '{parent}'"),
                        )
                    })
                })
                .transpose()?,
        );
    }
    // Every walk is bounded by 64 parent edges, regardless of authoring order.
    // Keep full matrices rather than decomposing a sheared parent/child product.
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
                return Err(SceneError::new(
                    "hierarchy_cycle",
                    format!("{}.parent", entries[index].1),
                    "parent references form a cycle",
                ));
            }
            if path.len() > MAX_DEPTH {
                return Err(SceneError::new(
                    "hierarchy_depth",
                    format!("{}.parent", entries[start].1),
                    "hierarchies allow at most 64 parent edges",
                ));
            }
            path.push(index);
            cursor = parents[index];
        }
        while let Some(index) = path.pop() {
            let (parent_matrix, depth) = parents[index].map_or((Mat4::IDENTITY, 0), |parent| {
                (
                    worlds[parent].expect("parent resolved before child"),
                    depths[parent] + 1,
                )
            });
            if depth > MAX_DEPTH {
                return Err(SceneError::new(
                    "hierarchy_depth",
                    format!("{}.parent", entries[index].1),
                    "hierarchies allow at most 64 parent edges",
                ));
            }
            let local = &transforms[index];
            let matrix = parent_matrix
                * Mat4::from_scale_rotation_translation(
                    local.scale,
                    local.rotation,
                    local.translation,
                );
            if !valid_global_matrix(matrix) {
                return Err(SceneError::new(
                    "invalid_world_transform",
                    format!("{}.transform", entries[index].1),
                    format!(
                        "composed transform for '{}' must be finite and invertible, with finite normal data and normalized normal-matrix inverse infinity norm <= 10000; reduce extreme anisotropy or shear",
                        entries[index].0.id
                    ),
                ));
            }
            if let Some(collider) = entries[index].0.collider {
                collider.validate_world(matrix, &format!("{}.collider", entries[index].1))?;
            }
            worlds[index] = Some(matrix);
            depths[index] = depth;
        }
    }
    Ok(())
}
