use engine_settings::{ErrorCode, Expected};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bevy_ecs::entity::Entity;
use engine_core::{Error, Mesh, ScenePath, Texture, Transform};

use crate::{
    MeshSource, SceneDocument, SceneFile, SceneFileKind, SceneFiles, SceneKinds,
    hierarchy::ExpandedObject, material::PreparedMaterial,
};

pub(crate) struct PreparedScene {
    pub environment: Option<crate::SceneEnvironment>,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<PreparedMaterial>,
    pub textures: Vec<Texture>,
    pub transforms: Vec<Transform>,
    pub entities: Vec<ExpandedObject>,
    pub paths: Vec<ScenePath>,
}

struct CpuAssets {
    textures: Vec<Texture>,
    meshes: Vec<Mesh>,
    materials: Vec<PreparedMaterial>,
}

impl SceneDocument {
    pub fn validate(&self, kinds: &SceneKinds) -> Result<(), Error> {
        self.prepare(None, kinds).map(|_| ())
    }

    pub fn validate_with(&self, kinds: &SceneKinds, files: &SceneFiles) -> Result<(), Error> {
        self.prepare(Some(files), kinds).map(|_| ())
    }

    pub(crate) fn structure(&self) -> Result<(), Error> {
        self.prepare(None, &SceneKinds::default()).map(|_| ())
    }

    pub fn files(&self, kinds: &SceneKinds) -> Vec<SceneFile> {
        let mut files: Vec<SceneFile> = Vec::new();
        let mut add = |uri: &String, kind| {
            if !files.iter().any(|file| file.uri == *uri) {
                files.push(SceneFile {
                    uri: uri.clone(),
                    kind,
                });
            }
        };
        for texture in &self.textures {
            if let Some(uri) = &texture.uri {
                add(uri, SceneFileKind::Image);
            }
        }
        for mesh in &self.meshes {
            match &mesh.source {
                MeshSource::Buffer { uri } => add(uri, SceneFileKind::Buffer),
                MeshSource::Cube { .. }
                | MeshSource::Plane { .. }
                | MeshSource::Octahedron { .. }
                | MeshSource::Inline { .. }
                | MeshSource::Provided { .. } => {}
            }
        }
        let entities = self
            .entities
            .iter()
            .chain(self.prefabs.iter().flat_map(|prefab| &prefab.entities));
        let cx = kinds.cx(None, None);
        crate::loading::named(self).lenient(|| {
            for entity in entities {
                for uri in kinds.files(entity, &cx) {
                    add(&uri, SceneFileKind::Buffer);
                }
            }
        });
        files
    }

    pub(crate) fn prepare(
        &self,
        files: Option<&SceneFiles>,
        kinds: &SceneKinds,
    ) -> Result<PreparedScene, Error> {
        identifier(&self.id, "$.id")?;
        self.validate_camera()?;
        self.validate_views()?;
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
            for (field, texture) in material.texture_ids() {
                reference(
                    texture,
                    &texture_ids,
                    &format!("$.materials[{index}].{field}"),
                )?;
            }
        }
        let named = crate::loading::named(self);
        let expanded = named.lenient(|| self.expand(&mesh_ids, &material_ids, kinds))?;
        let paths = crate::loading::paths(self, kinds)?;
        let entities = expanded
            .entities
            .iter()
            .map(|entity| (entity.source.id.clone(), Entity::PLACEHOLDER))
            .collect();
        let context = crate::loading::context(
            self,
            (named, BTreeMap::new()),
            paths.clone(),
            Arc::new(entities),
            kinds,
            files.map(SceneFiles::buffers),
        );
        if let (Some(section), Some(value)) = (kinds.streamed(), self.streamed_section()) {
            section.check(&value, "$")?;
            section.check_scene(&value, "$", &context)?;
        }
        let cx = kinds.cx(Some(&context), None);
        context.scope(|| {
            expanded
                .entities
                .iter()
                .try_for_each(|entity| kinds.prepare(&entity.source, &entity.path, &cx))
        })?;
        let CpuAssets {
            textures,
            meshes,
            materials,
        } = self.prepare_assets(files)?;
        Ok(PreparedScene {
            environment: self.environment.clone(),
            meshes,
            materials,
            textures,
            transforms: expanded.transforms,
            entities: expanded.entities,
            paths,
        })
    }

    fn prepare_assets(&self, files: Option<&SceneFiles>) -> Result<CpuAssets, Error> {
        for (mesh_index, mesh) in self.meshes.iter().enumerate() {
            if let MeshSource::Inline { vertices, .. } = &mesh.source {
                let path = format!("$.meshes[{mesh_index}].source");
                for (index, vertex) in vertices.iter().enumerate() {
                    bounded_vector(
                        vertex.position,
                        &format!("{path}.vertices[{index}].position"),
                        1_000_000.0,
                    )?;
                }
            }
        }
        let textures = self
            .textures
            .iter()
            .enumerate()
            .map(|(index, texture)| texture.texture(&format!("$.textures[{index}]"), files))
            .collect::<Result<Vec<_>, _>>()?;
        let meshes = self
            .meshes
            .iter()
            .enumerate()
            .map(|(index, mesh)| {
                mesh.source
                    .mesh(&format!("$.meshes[{index}].source"), files)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let materials = self
            .materials
            .iter()
            .enumerate()
            .map(|(index, material)| material.prepare(&format!("$.materials[{index}]")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CpuAssets {
            textures,
            meshes,
            materials,
        })
    }

    fn validate_camera(&self) -> Result<(), Error> {
        self.camera
            .validate()
            .map_err(|invalid| invalid.error(ErrorCode::InvalidCamera, "$.camera"))
    }
}

pub fn identifier(id: &str, path: &str) -> Result<(), Error> {
    if engine_settings::is_id(id) {
        return Ok(());
    }
    Err(Error::new(ErrorCode::InvalidId, path)
        .expected(Expected::Id)
        .at_most(engine_settings::MAX_ID_BYTES as u64)
        .got(id))
}

pub(crate) const MAX_NAME_BYTES: usize = 128;

pub(crate) fn display_name(name: &str, path: &str) -> Result<(), Error> {
    if name.len() > MAX_NAME_BYTES || name.chars().any(char::is_control) {
        return Err(Error::new(ErrorCode::InvalidName, path)
            .at_most(MAX_NAME_BYTES as u64)
            .got(name));
    }
    Ok(())
}

pub(crate) fn unique_ids<'a>(
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

pub(crate) fn reference(id: &str, known: &BTreeSet<&str>, path: &str) -> Result<(), Error> {
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

pub fn unknown<S: Into<String>>(
    error: Error,
    id: &str,
    known: impl IntoIterator<Item = S>,
) -> Error {
    let known: Vec<String> = known.into_iter().map(Into::into).collect();
    let error = error.got(id);
    match engine_settings::closest(id, &known) {
        Some(nearest) => error.nearest(nearest),
        None => error,
    }
}

pub(crate) fn moved(error: Error, code: ErrorCode, path: impl Into<String>) -> Error {
    Error {
        code,
        path: path.into(),
        detail: error.detail,
    }
}

fn bounded_vector(value: [f32; 3], path: &str, maximum: f32) -> Result<(), Error> {
    for (index, coordinate) in value.iter().enumerate() {
        if !coordinate.is_finite() || coordinate.abs() > maximum {
            return Err(
                Error::new(ErrorCode::InvalidNumber, format!("{path}[{index}]"))
                    .expected(Expected::Finite)
                    .range(-f64::from(maximum), f64::from(maximum))
                    .got(coordinate),
            );
        }
    }
    Ok(())
}
