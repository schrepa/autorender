use std::collections::BTreeSet;

use engine_core::{Material, Mesh, Texture, Transform};

use crate::{
    MeshSource, SCENE_VERSION, SceneDocument, SceneError, SceneTransform, hierarchy::ExpandedObject,
};

pub(crate) struct PreparedScene {
    pub colliders: Vec<Option<engine_collision::Collider>>,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    pub transforms: Vec<Transform>,
    pub entities: Vec<ExpandedObject>,
}

impl SceneDocument {
    /// Validates a document assembled through Rust or an authoring editor.
    ///
    /// Validation is independent of a world and may allocate temporary CPU
    /// geometry. Asset residency budgets are checked during instantiation.
    pub fn validate(&self) -> Result<(), SceneError> {
        self.prepare().map(|_| ())
    }

    pub(crate) fn prepare(&self) -> Result<PreparedScene, SceneError> {
        if !(1..=SCENE_VERSION).contains(&self.version) {
            return Err(SceneError::new(
                "unsupported_version",
                "$.version",
                format!(
                    "expected version 1 through {SCENE_VERSION}; found {}",
                    self.version
                ),
            ));
        }
        if self.version < 4
            && self
                .entities
                .iter()
                .chain(self.prefabs.iter().flat_map(|prefab| &prefab.entities))
                .any(|entity| entity.collider.is_some())
        {
            return Err(SceneError::new(
                "version_required",
                "$.version",
                "colliders require scene version 4; set version to 4 and increment revision",
            ));
        }
        if self.version == 1 && (!self.textures.is_empty() || self.materials.iter().any(|material| material.texture.is_some())
            || self.meshes.iter().any(|mesh| matches!(&mesh.source, MeshSource::Inline {vertices,..} if vertices.iter().any(|vertex| vertex.uv != [0.0;2])))) {
            return Err(SceneError::new("version_required", "$.version", "textures and UV data require scene version 2; set version to 2 and increment revision"));
        }
        if self.version < 3
            && (!self.prefabs.is_empty()
                || self.entities.iter().any(|entity| {
                    entity.parent.is_some()
                        || entity.prefab.is_some()
                        || entity.mesh.is_none()
                        || entity.material.is_none()
                }))
        {
            return Err(SceneError::new(
                "version_required",
                "$.version",
                "hierarchies, prefabs and transform-only nodes require scene version 3; set version to 3 and increment revision",
            ));
        }
        identifier(&self.id, "$.id")?;
        count(self.entities.len(), 10_000, "$.entities")?;
        count(self.meshes.len(), 128, "$.meshes")?;
        count(self.materials.len(), 256, "$.materials")?;
        count(self.textures.len(), 64, "$.textures")?;
        self.validate_camera()?;
        let mesh_ids = unique_ids(self.meshes.iter().map(|mesh| mesh.id.as_str()), "$.meshes")?;
        let material_ids = unique_ids(
            self.materials.iter().map(|material| material.id.as_str()),
            "$.materials",
        )?;
        let texture_ids = unique_ids(
            self.textures.iter().map(|texture| texture.id.as_str()),
            "$.textures",
        )?;
        for (index, material) in self.materials.iter().enumerate() {
            if let Some(texture) = &material.texture {
                reference(
                    texture,
                    &texture_ids,
                    &format!("$.materials[{index}].texture"),
                )?;
            }
        }
        let expanded = self.expand(&mesh_ids, &material_ids)?;
        let colliders = expanded
            .entities
            .iter()
            .map(|entity| {
                entity
                    .source
                    .collider
                    .map(|collider| collider.runtime(&format!("{}.collider", entity.path)))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Build every CPU asset before acquiring a mutable world resource.
        // Invalid late assets cannot leave earlier ones resident.
        for (mesh_index, mesh) in self.meshes.iter().enumerate() {
            if let MeshSource::Inline { vertices, indices } = &mesh.source {
                let path = format!("$.meshes[{mesh_index}].source");
                count(vertices.len(), 32_768, &format!("{path}.vertices"))?;
                count(indices.len(), 98_304, &format!("{path}.indices"))?;
                for (index, vertex) in vertices.iter().enumerate() {
                    bounded_vector(
                        vertex.position,
                        &format!("{path}.vertices[{index}].position"),
                        1_000_000.0,
                    )?;
                }
            }
        }
        let encoded_bytes = self
            .textures
            .iter()
            .try_fold(0usize, |sum, texture| sum.checked_add(texture.rgba8.len()));
        if encoded_bytes.is_none_or(|bytes| bytes > crate::document::MAX_SOURCE_BYTES) {
            return Err(SceneError::new(
                "source_too_large",
                "$.textures",
                "encoded texture payload must fit the 1 MiB source budget",
            ));
        }
        let textures = self
            .textures
            .iter()
            .enumerate()
            .map(|(index, texture)| texture.texture(&format!("$.textures[{index}]")))
            .collect::<Result<Vec<_>, _>>()?;
        let meshes = self
            .meshes
            .iter()
            .enumerate()
            .map(|(index, mesh)| mesh.source.mesh(&format!("$.meshes[{index}].source")))
            .collect::<Result<Vec<_>, _>>()?;
        let materials = self
            .materials
            .iter()
            .enumerate()
            .map(|(index, material)| material.material(&format!("$.materials[{index}].color")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PreparedScene {
            colliders,
            meshes,
            materials,
            textures,
            transforms: expanded.transforms,
            entities: expanded.entities,
        })
    }

    fn validate_camera(&self) -> Result<(), SceneError> {
        bounded_vector(self.camera.position, "$.camera.position", 1_000_000.0)?;
        bounded_vector(self.camera.target, "$.camera.target", 1_000_000.0)?;
        let direction = self.camera.camera().target - self.camera.camera().position;
        if direction.length_squared() < 0.000_001 {
            return Err(SceneError::new(
                "invalid_camera",
                "$.camera.target",
                "camera target must be at least 0.001 units from its position",
            ));
        }
        if !self.camera.fov_degrees.is_finite() || !(1.0..=179.0).contains(&self.camera.fov_degrees)
        {
            return Err(SceneError::new(
                "invalid_camera",
                "$.camera.fov_degrees",
                "vertical field of view must be finite and between 1 and 179 degrees",
            ));
        }
        if !self.camera.near.is_finite() || !(0.001..1_000_000.0).contains(&self.camera.near) {
            return Err(SceneError::new(
                "invalid_camera",
                "$.camera.near",
                "near plane must be finite and at least 0.001 but below 1000000",
            ));
        }
        if !self.camera.far.is_finite()
            || self.camera.far <= self.camera.near
            || self.camera.far - self.camera.near < 0.001
            || self.camera.far > 1_000_000.0
        {
            return Err(SceneError::new(
                "invalid_camera",
                "$.camera.far",
                "far plane must be finite, at least 0.001 beyond near, and at most 1000000",
            ));
        }
        Ok(())
    }
}

pub(crate) fn count(actual: usize, maximum: usize, path: &str) -> Result<(), SceneError> {
    if actual > maximum {
        return Err(SceneError::new(
            "count_limit",
            path,
            format!("at most {maximum} entries are allowed; found {actual}"),
        ));
    }
    Ok(())
}

pub(crate) fn identifier(id: &str, path: &str) -> Result<(), SceneError> {
    if id.is_empty()
        || id.len() > 64
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(SceneError::new(
            "invalid_id",
            path,
            "IDs must contain 1 to 64 ASCII letters, digits, underscores, hyphens or periods",
        ));
    }
    Ok(())
}

pub(crate) fn unique_ids<'a>(
    ids: impl Iterator<Item = &'a str>,
    collection_path: &str,
) -> Result<BTreeSet<&'a str>, SceneError> {
    let mut known = BTreeSet::new();
    for (index, id) in ids.enumerate() {
        let path = format!("{collection_path}[{index}].id");
        identifier(id, &path)?;
        if !known.insert(id) {
            return Err(SceneError::new(
                "duplicate_id",
                path,
                format!("authored ID '{id}' is already used in this collection"),
            ));
        }
    }
    Ok(known)
}

pub(crate) fn reference(id: &str, known: &BTreeSet<&str>, path: &str) -> Result<(), SceneError> {
    identifier(id, path)?;
    if !known.contains(id) {
        return Err(SceneError::new(
            "unknown_reference",
            path,
            format!("no authored object in this collection has ID '{id}'"),
        ));
    }
    Ok(())
}

fn bounded_vector(value: [f32; 3], path: &str, maximum: f32) -> Result<(), SceneError> {
    for (index, coordinate) in value.iter().enumerate() {
        if !coordinate.is_finite() || coordinate.abs() > maximum {
            return Err(SceneError::new(
                "invalid_number",
                format!("{path}[{index}]"),
                format!("value must be finite and within ±{maximum}"),
            ));
        }
    }
    Ok(())
}

pub(crate) fn transform(source: &SceneTransform, path: &str) -> Result<Transform, SceneError> {
    bounded_vector(
        source.translation,
        &format!("{path}.translation"),
        1_000_000.0,
    )?;
    bounded_vector(
        source.rotation_degrees,
        &format!("{path}.rotation_degrees"),
        360_000.0,
    )?;
    for (index, scale) in source.scale.iter().enumerate() {
        if !scale.is_finite() || !(0.0001..=10_000.0).contains(scale) {
            return Err(SceneError::new(
                "invalid_scale",
                format!("{path}.scale[{index}]"),
                "scale must be finite and between 0.0001 and 10000 on every axis",
            ));
        }
    }
    Ok(source.transform())
}
