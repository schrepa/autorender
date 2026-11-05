use engine_settings::ErrorCode;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::{World, WorldId};
use engine_assets::SceneAssets;
use engine_core::{
    AssetLibrary, Error, EvenDisabled, MaterialHandle, Mesh3d, MeshHandle, ProvidedAssets,
    SceneContext, TextureHandle, Transform, propagate_transforms, set_parents,
};
use engine_registry::{OperationScene, SceneEntity};

use crate::{
    SceneDocument, SceneFiles, SceneKinds, hierarchy::ExpandedObject, validation::PreparedScene,
};

#[derive(Debug)]
#[must_use = "retain the instance and explicitly despawn it when the scene is disposed"]
pub struct SceneInstance {
    id: String,
    revision: u64,
    world_id: WorldId,
    entities: Arc<BTreeMap<String, Entity>>,
    assets: Handles,
    owned: Handles,
    material_ids: Vec<String>,
    mesh_ids: Vec<String>,
    texture_ids: Vec<String>,
    origins: Origins,
    skipped: Vec<Error>,
}

#[derive(Debug, Default)]
struct Origins {
    meshes: Vec<MeshOrigin>,
    textures: Vec<TextureOrigin>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MeshOrigin {
    Cube,
    Plane,
    Octahedron,
    Inline { vertices: usize, indices: usize },
    Buffer { uri: String },
    Provided { name: String },
}

impl From<&crate::MeshSource> for MeshOrigin {
    fn from(source: &crate::MeshSource) -> Self {
        match source {
            crate::MeshSource::Cube {} => Self::Cube,
            crate::MeshSource::Plane {} => Self::Plane,
            crate::MeshSource::Octahedron {} => Self::Octahedron,
            crate::MeshSource::Inline { vertices, indices } => Self::Inline {
                vertices: vertices.len(),
                indices: indices.len(),
            },
            crate::MeshSource::Buffer { uri } => Self::Buffer { uri: uri.clone() },
            crate::MeshSource::Provided { name } => Self::Provided { name: name.clone() },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TextureOrigin {
    Inline,
    File { uri: String },
    Provided { name: String },
}

impl From<&crate::SceneTexture> for TextureOrigin {
    fn from(texture: &crate::SceneTexture) -> Self {
        match (&texture.provided, &texture.uri) {
            (Some(name), _) => Self::Provided { name: name.clone() },
            (None, Some(uri)) => Self::File { uri: uri.clone() },
            (None, None) => Self::Inline,
        }
    }
}

#[derive(Debug, Default)]
struct Handles {
    meshes: Vec<MeshHandle>,
    materials: Vec<MaterialHandle>,
    textures: Vec<TextureHandle>,
}

impl Handles {
    fn remove(&self, assets: &mut AssetLibrary) {
        for &handle in &self.meshes {
            assets.remove_mesh(handle);
        }
        for &handle in &self.materials {
            assets.remove_material(handle);
        }
        for &handle in &self.textures {
            assets.remove_texture(handle);
        }
    }
}

fn insert_each<T, H: Copy>(
    items: Vec<T>,
    provided: Vec<Option<H>>,
    (handles, owned): (&mut Vec<H>, &mut Vec<H>),
    field: &str,
    mut insert: impl FnMut(T) -> Result<H, Error>,
) -> Result<(), Error> {
    for (index, (item, provided)) in items.into_iter().zip(provided).enumerate() {
        let handle = match provided {
            Some(handle) => handle,
            None => {
                let handle = insert(item).map_err(|error| {
                    error
                        .coded(ErrorCode::AssetBudgetExceeded)
                        .at(format!("$.{field}[{index}]"))
                })?;
                owned.push(handle);
                handle
            }
        };
        handles.push(handle);
    }
    Ok(())
}

fn provided_each<'a, H, N>(
    names: impl Iterator<Item = Option<&'a str>>,
    path: impl Fn(usize) -> String,
    find: impl Fn(&str) -> Option<H>,
    known: impl Fn() -> N,
) -> Result<Vec<Option<H>>, Error>
where
    N: IntoIterator,
    N::Item: Into<String>,
{
    names
        .enumerate()
        .map(|(index, name)| {
            name.map(|name| {
                find(name).ok_or_else(|| {
                    Error::new(ErrorCode::UnknownReference, path(index))
                        .got(name)
                        .variants(known())
                })
            })
            .transpose()
        })
        .collect()
}

impl SceneInstance {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn entity(&self, authored_id: &str) -> Option<Entity> {
        self.entities.get(authored_id).copied()
    }

    pub fn authored_entity(&self, world: &World, id: &str) -> Option<Entity> {
        self.entity(id).or_else(|| {
            world
                .get_resource::<OperationScene>()
                .filter(|scene| Arc::ptr_eq(&scene.entities, &self.entities))?
                .entity(world, id)
        })
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn skipped(&self) -> &[Error] {
        &self.skipped
    }

    pub fn textures(&self) -> impl ExactSizeIterator<Item = (&str, TextureHandle)> {
        self.texture_ids
            .iter()
            .zip(&self.assets.textures)
            .map(|(id, handle)| (id.as_str(), *handle))
    }

    pub fn meshes(&self) -> impl ExactSizeIterator<Item = (&str, MeshHandle)> {
        self.mesh_ids
            .iter()
            .zip(&self.assets.meshes)
            .map(|(id, handle)| (id.as_str(), *handle))
    }

    pub fn materials(&self) -> impl ExactSizeIterator<Item = (&str, MaterialHandle)> {
        self.material_ids
            .iter()
            .zip(&self.assets.materials)
            .map(|(id, handle)| (id.as_str(), *handle))
    }

    pub fn mesh_origin(&self, handle: MeshHandle) -> Option<(&str, &MeshOrigin)> {
        let index = self.assets.meshes.iter().position(|&mesh| mesh == handle)?;
        Some((&self.mesh_ids[index], &self.origins.meshes[index]))
    }

    pub fn texture_origin(&self, handle: TextureHandle) -> Option<(&str, &TextureOrigin)> {
        let index = self
            .assets
            .textures
            .iter()
            .position(|&texture| texture == handle)?;
        Some((&self.texture_ids[index], &self.origins.textures[index]))
    }

    pub fn entities(&self) -> impl ExactSizeIterator<Item = (&str, Entity)> {
        self.entities
            .iter()
            .map(|(id, entity)| (id.as_str(), *entity))
    }

    pub fn operation_scene(&self) -> OperationScene {
        OperationScene {
            id: self.id.clone(),
            entities: self.entities.clone(),
            streamed: false,
        }
    }

    pub fn current_scene(&self, world: &World) -> OperationScene {
        world
            .get_resource::<OperationScene>()
            .filter(|scene| Arc::ptr_eq(&scene.entities, &self.entities))
            .cloned()
            .unwrap_or_else(|| self.operation_scene())
    }

    pub fn belongs_to(&self, world: &World) -> bool {
        self.world_id == world.id()
    }

    pub fn despawn(self, world: &mut World) {
        assert_eq!(self.world_id, world.id(), "scene_world_matches");
        if world
            .get_resource::<OperationScene>()
            .is_some_and(|scene| Arc::ptr_eq(&scene.entities, &self.entities))
        {
            world.remove_resource::<OperationScene>();
        }
        if world
            .get_resource::<SceneContext>()
            .is_some_and(|context| Arc::ptr_eq(&context.entities, &self.entities))
        {
            world.remove_resource::<SceneContext>();
        }
        self.remove_streamed(world);
        for &entity in self.entities.values() {
            world.despawn(entity);
        }
        if let Some(mut assets) = world.get_resource_mut::<AssetLibrary>() {
            self.owned.remove(&mut assets);
        }
    }

    fn remove_streamed(&self, world: &mut World) {
        let mut kept: BTreeSet<Entity> = self.entities.values().copied().collect();
        if let Some(current) = world.get_resource::<OperationScene>() {
            kept.extend(current.entities.values().copied());
        }
        let streamed: Vec<Entity> = world
            .query_filtered::<(Entity, &SceneEntity), EvenDisabled>()
            .iter(world)
            .filter(|(entity, authored)| authored.scene_id == self.id && !kept.contains(entity))
            .map(|(entity, _)| entity)
            .collect();
        for entity in streamed {
            let _ = world.try_despawn(entity);
        }
    }
}

impl SceneDocument {
    pub fn instantiate(&self, world: &mut World) -> Result<SceneInstance, Error> {
        self.instantiate_with(world, &SceneFiles::default())
    }

    pub fn instantiate_with(
        &self,
        world: &mut World,
        files: &SceneFiles,
    ) -> Result<SceneInstance, Error> {
        let kinds = SceneKinds::of(world);
        let mut prepared = self.prepare(Some(files), &kinds)?;
        let skipped = crate::loading::named(self).scope(|| self.admit(world, &kinds))?;
        let chose_left_out = world
            .get_resource::<engine_registry::Registry>()
            .is_some_and(|registry| !registry.left_out().is_empty());
        if let Some(first) = skipped.first().filter(|_| !chose_left_out) {
            return Err(first.clone());
        }
        let bounds = crate::loading::bounds(self, &prepared.meshes);
        let (assets, owned) = self.insert_assets(world, &mut prepared)?;
        let table = Arc::new(self.table(&assets));
        let spawned = self.spawn(world, &prepared, &table);
        let entities: BTreeMap<String, Entity> = prepared
            .entities
            .iter()
            .zip(&spawned)
            .map(|(expanded, &entity)| (expanded.source.id.clone(), entity))
            .collect();
        let entities = Arc::new(entities);
        let context = crate::loading::context(
            self,
            (table, bounds),
            std::mem::take(&mut prepared.paths),
            entities.clone(),
            &kinds,
            Some(files.buffers()),
        );
        if let Err(error) = read(world, &prepared, &spawned, &context, &kinds) {
            for &entity in &spawned {
                world.despawn(entity);
            }
            owned.remove(&mut world.resource_mut::<AssetLibrary>());
            return Err(error);
        }
        let parents: Vec<_> = prepared
            .entities
            .iter()
            .filter_map(|expanded| {
                let source = &expanded.source;
                source.parent.as_ref().map(|parent| {
                    (
                        entities[source.id.as_str()],
                        Some(entities[parent.as_str()]),
                    )
                })
            })
            .collect();
        if let Err(error) = set_parents(world, &parents) {
            for &entity in entities.values() {
                world.despawn(entity);
            }
            owned.remove(&mut world.resource_mut::<AssetLibrary>());
            return Err(error
                .coded(ErrorCode::HierarchyInstallFailed)
                .within("$.entities"));
        }
        propagate_transforms(world);
        world.insert_resource(self.camera.camera());
        world.insert_resource(SceneContext {
            files: None,
            ..context.clone()
        });
        if let Some(environment) = &prepared.environment {
            context.scope(|| environment.apply_all(world, "$.environment"))?;
        }
        world.insert_resource(crate::SceneViews::of(self));
        context.scope(|| crate::environment::apply_streamed(world, self.streamed_section()))?;
        let instance = SceneInstance {
            id: self.id.clone(),
            revision: self.revision,
            world_id: world.id(),
            entities,
            assets,
            owned,
            material_ids: self
                .materials
                .iter()
                .map(|material| material.id.clone())
                .collect(),
            mesh_ids: self.meshes.iter().map(|mesh| mesh.id.clone()).collect(),
            texture_ids: self
                .textures
                .iter()
                .map(|texture| texture.id.clone())
                .collect(),
            origins: Origins {
                meshes: self
                    .meshes
                    .iter()
                    .map(|mesh| MeshOrigin::from(&mesh.source))
                    .collect(),
                textures: self.textures.iter().map(TextureOrigin::from).collect(),
            },
            skipped,
        };
        world.insert_resource(instance.operation_scene());
        Ok(instance)
    }

    fn insert_assets(
        &self,
        world: &mut World,
        prepared: &mut PreparedScene,
    ) -> Result<(Handles, Handles), Error> {
        let none = ProvidedAssets::default();
        let provided = world.get_resource::<ProvidedAssets>().unwrap_or(&none);
        let provided_meshes = provided_each(
            self.meshes.iter().map(|mesh| mesh.source.provided()),
            |index| format!("$.meshes[{index}].source.name"),
            |name| provided.mesh(name),
            || provided.mesh_names(),
        )?;
        let provided_textures = provided_each(
            self.textures
                .iter()
                .map(|texture| texture.provided.as_deref()),
            |index| format!("$.textures[{index}].provided"),
            |name| provided.texture(name),
            || provided.texture_names(),
        )?;
        let provided_materials = provided_each(
            self.materials
                .iter()
                .map(|material| material.provided.as_deref()),
            |index| format!("$.materials[{index}].provided"),
            |name| provided.material(name),
            || provided.material_names(),
        )?;
        let mut handles = Handles::default();
        let mut owned = Handles::default();
        let mut assets = world
            .get_resource_mut::<AssetLibrary>()
            .ok_or_else(|| Error::new(ErrorCode::MissingAssetLibrary, "$"))?;
        let inserted = (|| {
            insert_each(
                std::mem::take(&mut prepared.meshes),
                provided_meshes,
                (&mut handles.meshes, &mut owned.meshes),
                "meshes",
                |mesh| assets.insert_mesh(mesh),
            )?;
            insert_each(
                std::mem::take(&mut prepared.textures),
                provided_textures,
                (&mut handles.textures, &mut owned.textures),
                "textures",
                |texture| assets.insert_texture(texture),
            )?;
            let texture_lookup: BTreeMap<_, _> = self
                .textures
                .iter()
                .zip(&handles.textures)
                .map(|(source, handle)| (source.id.as_str(), *handle))
                .collect();
            insert_each(
                std::mem::take(&mut prepared.materials),
                provided_materials,
                (&mut handles.materials, &mut owned.materials),
                "materials",
                |material| assets.insert_material(material.material(|id| texture_lookup[id])),
            )
        })();
        if let Err(error) = inserted {
            owned.remove(&mut assets);
            return Err(error);
        }
        Ok((handles, owned))
    }

    fn spawn(
        &self,
        world: &mut World,
        prepared: &PreparedScene,
        table: &SceneAssets,
    ) -> Vec<Entity> {
        prepared
            .entities
            .iter()
            .zip(&prepared.transforms)
            .map(|(expanded, &transform)| {
                spawn_entity(world, (self.id.as_str(), table), expanded, transform)
            })
            .collect()
    }
}

impl SceneDocument {
    fn table(&self, handles: &Handles) -> SceneAssets {
        SceneAssets {
            meshes: self
                .meshes
                .iter()
                .zip(&handles.meshes)
                .map(|(source, handle)| (source.id.clone(), *handle))
                .collect(),
            materials: self
                .materials
                .iter()
                .zip(&handles.materials)
                .map(|(source, handle)| (source.id.clone(), *handle))
                .collect(),
            textures: self
                .textures
                .iter()
                .zip(&handles.textures)
                .map(|(source, handle)| (source.id.clone(), *handle))
                .collect(),
        }
    }
}

fn read(
    world: &mut World,
    prepared: &PreparedScene,
    spawned: &[Entity],
    context: &SceneContext,
    kinds: &SceneKinds,
) -> Result<(), Error> {
    let cx = kinds.cx(Some(context), None);
    context.scope(|| {
        prepared
            .entities
            .iter()
            .zip(spawned)
            .try_for_each(|(expanded, &entity)| {
                kinds.read(world, entity, &expanded.source, &expanded.path, &cx)
            })
    })
}

fn spawn_entity(
    world: &mut World,
    (scene_id, table): (&str, &SceneAssets),
    expanded: &ExpandedObject,
    transform: Transform,
) -> Entity {
    let source = &expanded.source;
    let mut entity = world.spawn((
        transform,
        SceneEntity {
            scene_id: scene_id.to_owned(),
            authored_id: source.id.clone(),
            name: source.name.clone(),
            tags: source.tags.clone(),
            authored_parent: source.parent.clone(),
            prefab_id: expanded.prefab_id.clone(),
            prefab_instance: expanded.prefab_instance.clone(),
            prefab_local_id: expanded.prefab_local_id.clone(),
        },
    ));
    if let (Some(mesh), Some(material)) = (&source.mesh, &source.material) {
        entity.insert(Mesh3d {
            mesh: table.meshes[mesh.as_str()],
            material: table.materials[material.as_str()],
        });
    }
    entity.id()
}
