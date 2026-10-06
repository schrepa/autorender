use std::collections::{BTreeMap, BTreeSet};

use engine_core::{MAX_HIERARCHY_DEPTH as MAX_DEPTH, Transform, glam::Mat4, valid_global_matrix};

use crate::validation::{identifier, reference, transform, unique_ids};
use crate::{SceneDocument, SceneError, SceneObject, ScenePrefab};

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

/// A document's prefab definitions by ID, with their source indices.
type Prefabs<'a> = BTreeMap<&'a str, (usize, &'a ScenePrefab)>;

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

    /// The document's entities and its streamed entities (decision 086),
    /// each list expanded and validated on its own: parents stay within
    /// a list, and expanded IDs are unique across both.
    pub(crate) fn expand(
        &self,
        mesh_ids: &BTreeSet<&str>,
        material_ids: &BTreeSet<&str>,
    ) -> Result<[ExpandedScene; 2], SceneError> {
        let (prefab_ids, prefabs) = self.validate_prefabs(mesh_ids, material_ids)?;
        let ids = (mesh_ids, material_ids, &prefab_ids);
        let entities = expand_list(&self.entities, "$.entities", ids, &prefabs)?;
        let streamed = expand_list(&self.streamed, "$.streamed", ids, &prefabs)?;
        let known: BTreeSet<&str> = entities
            .entities
            .iter()
            .map(|entity| entity.source.id.as_str())
            .collect();
        for entity in &streamed.entities {
            if known.contains(entity.source.id.as_str()) {
                return Err(SceneError::new(
                    "duplicate_id",
                    format!("{}.id", entity.path),
                    format!(
                        "expanded entity ID '{}' is already used by an entity",
                        entity.source.id
                    ),
                ));
            }
        }
        Ok([entities, streamed])
    }

    /// Validates prefab definitions: their IDs, entities and local graphs.
    /// Returns their IDs and the definitions by ID.
    fn validate_prefabs(
        &self,
        mesh_ids: &BTreeSet<&str>,
        material_ids: &BTreeSet<&str>,
    ) -> Result<(BTreeSet<&str>, Prefabs<'_>), SceneError> {
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

        let prefab_lookup: Prefabs<'_> = self
            .prefabs
            .iter()
            .enumerate()
            .map(|(index, prefab)| (prefab.id.as_str(), (index, prefab)))
            .collect();
        Ok((prefab_ids, prefab_lookup))
    }
}

/// `list` (at authored path `list_path`) with prefab placements expanded,
/// validated against the document's mesh, material and prefab IDs.
fn expand_list(
    list: &[SceneObject],
    list_path: &str,
    (mesh_ids, material_ids, prefab_ids): (&BTreeSet<&str>, &BTreeSet<&str>, &BTreeSet<&str>),
    prefab_lookup: &Prefabs<'_>,
) -> Result<ExpandedScene, SceneError> {
    unique_ids(list.iter().map(|entity| entity.id.as_str()), list_path)?;
    let mut expanded_count = list.len();
    for (index, entity) in list.iter().enumerate() {
        let path = format!("{list_path}[{index}]");
        validate_object(entity, &path, mesh_ids, material_ids, prefab_ids)?;
        if let Some(id) = &entity.prefab {
            expanded_count += prefab_lookup[id.as_str()].1.entities.len();
        }
    }
    let mut entities = Vec::with_capacity(expanded_count);
    let mut transforms = Vec::with_capacity(expanded_count);
    for (index, entity) in list.iter().enumerate() {
        let mut source = entity.clone();
        source.prefab = None;
        entities.push(ExpandedObject {
            source,
            path: format!("{list_path}[{index}]"),
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
        // Agents (decision 089) walk unparented.
        if entity.source.nav_agent.is_some() && entity.source.parent.is_some() {
            return Err(SceneError::new(
                "invalid_nav_agent",
                format!("{}.nav_agent", entity.path),
                "only roots walk; give the placement or root the agent instead",
            ));
        }
        // Grounded objects (decision 086) stand on the ground
        // themselves.
        if entity.source.grounded.is_some() && entity.source.parent.is_some() {
            return Err(SceneError::new(
                "invalid_grounded",
                format!("{}.grounded", entity.path),
                "only roots stand on the ground; ground the placement or root instead",
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
    if let Some(decal) = &entity.decal {
        reference(
            &decal.material,
            material_ids,
            &format!("{path}.decal.material"),
        )?;
    }
    if let Some(agent) = entity.nav_agent {
        agent.validate().map_err(|invalid| {
            SceneError::new(
                "invalid_nav_agent",
                format!("{path}.nav_agent.{}", invalid.field),
                invalid.rule,
            )
        })?;
    }
    if let Some(deformer) = entity.deformer {
        deformer.validate().map_err(|invalid| {
            SceneError::new(
                "invalid_deformer",
                format!("{path}.deformer.{}", invalid.field),
                invalid.rule,
            )
        })?;
    }
    if let Some(spline) = &entity.spline {
        spline.validate(&format!("{path}.spline"))?;
        reference(&spline.mesh, mesh_ids, &format!("{path}.spline.mesh"))?;
        reference(
            &spline.material,
            material_ids,
            &format!("{path}.spline.material"),
        )?;
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
