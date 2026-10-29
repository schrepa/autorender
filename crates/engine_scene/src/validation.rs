use engine_core::collections::HashMap;
use engine_settings::ErrorCode;
use std::collections::{BTreeMap, BTreeSet};

use engine_core::{
    Error, Expected, Mesh, Texture, Transform,
    glam::{Mat4, Vec3},
};

use crate::{
    MeshSource, SceneDocument, SceneFile, SceneFileKind, SceneFiles,
    components::{PreparedComponents, Reach},
    hierarchy::{ExpandedObject, ExpandedScene},
    material::PreparedMaterial,
};

pub(crate) struct PreparedScene {
    pub environment: Option<crate::SceneEnvironment>,
    pub components: Vec<PreparedEntity>,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<PreparedMaterial>,
    pub textures: Vec<Texture>,
    pub transforms: Vec<Transform>,
    pub entities: Vec<ExpandedObject>,
    pub streamed: PreparedStreamed,
}

#[derive(Default)]
pub(crate) struct PreparedStreamed {
    pub entities: Vec<ExpandedObject>,
    pub transforms: Vec<Transform>,
    pub components: Vec<PreparedEntity>,
    pub parents: Vec<Option<usize>>,
    pub items: Vec<PreparedItem>,
}

pub(crate) struct PreparedItem {
    pub entities: Vec<usize>,
    pub position: Vec3,
    pub reach: f32,
}

pub(crate) struct PreparedEntity {
    pub components: PreparedComponents,
}

struct CpuAssets {
    textures: Vec<Texture>,
    meshes: Vec<Mesh>,
    materials: Vec<PreparedMaterial>,
}

impl SceneDocument {
    pub fn validate(&self) -> Result<(), Error> {
        self.prepare(None).map(|_| ())
    }

    pub fn validate_with(&self, files: &SceneFiles) -> Result<(), Error> {
        self.prepare(Some(files)).map(|_| ())
    }

    pub fn files(&self) -> Vec<SceneFile> {
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
                | MeshSource::Inline { .. } => {}
            }
        }
        let entities = self
            .entities
            .iter()
            .chain(self.prefabs.iter().flat_map(|prefab| &prefab.entities));
        for entity in entities {
            if let Some(terrain) = &entity.terrain {
                for uri in terrain.files() {
                    add(uri, SceneFileKind::Buffer);
                }
            }
        }
        files
    }

    pub(crate) fn prepare(&self, files: Option<&SceneFiles>) -> Result<PreparedScene, Error> {
        identifier(&self.id, "$.id")?;
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
            for (field, texture) in material.texture_ids() {
                reference(
                    texture,
                    &texture_ids,
                    &format!("$.materials[{index}].{field}"),
                )?;
            }
        }
        let [expanded, streamed] = self.expand(&mesh_ids, &material_ids)?;
        let components = expanded
            .entities
            .iter()
            .map(|entity| self.prepare_entity(entity, files))
            .collect::<Result<Vec<_>, _>>()?;

        let streamed_components = self.prepare_streamed(&streamed, files)?;

        let CpuAssets {
            textures,
            meshes,
            materials,
        } = self.prepare_assets(files)?;
        let streamed = self.streamed_items(streamed, streamed_components, &meshes);
        if let Some(environment) = &self.environment {
            environment.validate("$.environment")?;
        }
        Ok(PreparedScene {
            environment: self.environment.clone(),
            components,
            meshes,
            materials,
            textures,
            transforms: expanded.transforms,
            entities: expanded.entities,
            streamed,
        })
    }

    fn prepare_streamed(
        &self,
        streamed: &ExpandedScene,
        files: Option<&SceneFiles>,
    ) -> Result<Vec<PreparedEntity>, Error> {
        streamed
            .entities
            .iter()
            .map(|entity| {
                entity.source.check_streamed(&entity.path)?;
                self.prepare_entity(entity, files)
            })
            .collect()
    }

    fn streamed_items(
        &self,
        streamed: ExpandedScene,
        components: Vec<PreparedEntity>,
        meshes: &[Mesh],
    ) -> PreparedStreamed {
        let indices: BTreeMap<&str, usize> = streamed
            .entities
            .iter()
            .enumerate()
            .map(|(index, entity)| (entity.source.id.as_str(), index))
            .collect();
        let parents: Vec<Option<usize>> = streamed
            .entities
            .iter()
            .map(|entity| {
                entity
                    .source
                    .parent
                    .as_ref()
                    .map(|parent| indices[parent.as_str()])
            })
            .collect();
        let roots = item_roots(&streamed.entities, &indices);
        let mut worlds: Vec<Option<Mat4>> = vec![None; streamed.entities.len()];
        let mut chain = Vec::new();
        for start in 0..worlds.len() {
            let mut cursor = Some(start);
            while let Some(index) = cursor.filter(|&index| worlds[index].is_none()) {
                chain.push(index);
                cursor = parents[index];
            }
            while let Some(index) = chain.pop() {
                let local = streamed.transforms[index];
                let local = Mat4::from_scale_rotation_translation(
                    local.scale,
                    local.rotation,
                    local.translation,
                );
                worlds[index] = Some(parents[index].map_or(local, |parent| {
                    worlds[parent].expect("parent_composed_first") * local
                }));
            }
        }
        let mesh_indices: BTreeMap<&str, usize> = self
            .meshes
            .iter()
            .enumerate()
            .map(|(index, mesh)| (mesh.id.as_str(), index))
            .collect();
        let mut items: Vec<PreparedItem> = Vec::new();
        let mut item_of = HashMap::default();
        for (index, &root) in roots.iter().enumerate() {
            let item = *item_of.entry(root).or_insert_with(|| {
                items.push(PreparedItem {
                    entities: Vec::new(),
                    position: worlds[root]
                        .expect("world_matrix_composed")
                        .w_axis
                        .truncate(),
                    reach: 0.0,
                });
                items.len() - 1
            });
            items[item].entities.push(index);
            let position = items[item].position;
            let world = worlds[index].expect("world_matrix_composed");
            let from = world.w_axis.truncate().distance(position);
            let standing = Reach {
                world,
                position,
                from,
                meshes: (meshes, &mesh_indices),
            };
            let source = &streamed.entities[index].source;
            let component = &components[index];
            let mut reach = from;
            let mut extend = |value: f32| reach = engine_core::math::max(reach, value);
            if let Some(mesh) = source
                .mesh
                .as_ref()
                .and_then(|id| mesh_indices.get(id.as_str()))
            {
                let bounds = meshes[*mesh].bounds();
                extend(standing.of_box(bounds.min(), bounds.max()));
            }
            component.components.reach(&standing, &mut extend);
            if reach.is_finite() {
                items[item].reach = engine_core::math::max(items[item].reach, reach);
            }
        }
        for item in &mut items {
            item.entities
                .sort_by_key(|&index| (parents[index].is_some(), index));
        }
        PreparedStreamed {
            entities: streamed.entities,
            transforms: streamed.transforms,
            components,
            parents,
            items,
        }
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

    fn prepare_entity(
        &self,
        entity: &ExpandedObject,
        files: Option<&SceneFiles>,
    ) -> Result<PreparedEntity, Error> {
        let (source, path) = (&entity.source, &entity.path);
        Ok(PreparedEntity {
            components: source.prepare_components(path, self, files)?,
        })
    }

    fn validate_camera(&self) -> Result<(), Error> {
        self.camera
            .validate()
            .map_err(|invalid| invalid.error(ErrorCode::InvalidCamera, "$.camera"))
    }
}

fn item_roots(entities: &[ExpandedObject], indices: &BTreeMap<&str, usize>) -> Vec<usize> {
    let parent = |index: usize| {
        entities[index]
            .source
            .parent
            .as_ref()
            .map(|parent| indices[parent.as_str()])
    };
    (0..entities.len())
        .map(|mut index| {
            while let Some(next) = parent(index) {
                index = next;
            }
            index
        })
        .collect()
}

pub(crate) fn identifier(id: &str, path: &str) -> Result<(), Error> {
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

pub(crate) fn unknown<S: Into<String>>(
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
