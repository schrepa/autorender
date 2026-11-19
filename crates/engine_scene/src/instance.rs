use engine_settings::{ErrorCode, Expected};
use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use bevy_ecs::world::{World, WorldId};
use engine_assets::SceneAssets;
use engine_core::{
    AssetLibrary, Error, GeometryHandle, MaterialHandle, OperationScene, SceneBuffers,
    SceneContext, SceneCx, SceneEntity, TextureHandle, Transform, propagate_transforms,
    set_parents,
};
use engine_registry::SettingsEntry;
use serde_json::Value;

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
    Inline { vertices: usize, indices: usize },
    Buffer { uri: String },
}

impl From<&crate::MeshSource> for MeshOrigin {
    fn from(source: &crate::MeshSource) -> Self {
        match source {
            crate::MeshSource::Cube {} => Self::Cube,
            crate::MeshSource::Plane {} => Self::Plane,
            crate::MeshSource::Inline { vertices, indices } => Self::Inline {
                vertices: vertices.len(),
                indices: indices.len(),
            },
            crate::MeshSource::Buffer { uri } => Self::Buffer { uri: uri.clone() },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TextureOrigin {
    Inline,
    File { uri: String },
}

impl From<&crate::SceneTexture> for TextureOrigin {
    fn from(texture: &crate::SceneTexture) -> Self {
        match &texture.uri {
            Some(uri) => Self::File { uri: uri.clone() },
            None => Self::Inline,
        }
    }
}

#[derive(Debug, Default)]
struct Handles {
    meshes: Vec<GeometryHandle>,
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
    items: impl IntoIterator<Item = Option<T>>,
    handles: &mut Vec<H>,
    field: &str,
    mut insert: impl FnMut(T) -> Result<H, Error>,
) -> Result<(), Error> {
    for (index, item) in items.into_iter().enumerate() {
        let item = item.ok_or_else(|| {
            Error::new(ErrorCode::MissingFile, format!("$.{field}[{index}].source"))
                .expected(Expected::File)
        })?;
        let handle = insert(item).map_err(|error| {
            error
                .coded(ErrorCode::AssetBudgetExceeded)
                .at(format!("$.{field}[{index}]"))
        })?;
        handles.push(handle);
    }
    Ok(())
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

    pub fn materials(&self) -> impl ExactSizeIterator<Item = (&str, MaterialHandle)> {
        self.material_ids
            .iter()
            .zip(&self.assets.materials)
            .map(|(id, handle)| (id.as_str(), *handle))
    }

    pub fn mesh_origin(&self, handle: GeometryHandle) -> Option<(&str, &MeshOrigin)> {
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

    pub fn despawn(self, world: &mut World) {
        assert_eq!(self.world_id, world.id(), "scene_world_matches");
        let installed: Option<Vec<Entity>> = world
            .get_resource::<OperationScene>()
            .filter(|scene| Arc::ptr_eq(&scene.entities, &self.entities))
            .map(|scene| scene.installed.values().copied().collect());
        if installed.is_some() {
            world.remove_resource::<OperationScene>();
        }
        if world
            .get_resource::<SceneContext>()
            .is_some_and(|context| Arc::ptr_eq(&context.entities, &self.entities))
        {
            world.remove_resource::<SceneContext>();
        }
        for entity in installed.into_iter().flatten() {
            let _ = world.try_despawn(entity);
        }
        for &entity in self.entities.values() {
            let _ = world.try_despawn(entity);
        }
        if let Some(mut assets) = world.get_resource_mut::<AssetLibrary>() {
            self.assets.remove(&mut assets);
        }
    }
}

impl SceneDocument {
    pub fn instantiate_with(
        &self,
        world: &mut World,
        files: &SceneFiles,
    ) -> Result<SceneInstance, Error> {
        let kinds = SceneKinds::of(world);
        let mut prepared = self.prepare(Some(files), &kinds)?;
        let skipped = crate::loading::named(self).scope(|| self.admit(world, &kinds))?;
        if let Some(first) = skipped.first() {
            return Err(first.clone());
        }
        let replaced = Replaced::of(world);
        let assets = self.insert_assets(world, &mut prepared)?;
        let table = Arc::new(self.table(&assets));
        let spawned = self.spawn(world, &prepared);
        let entities: BTreeMap<String, Entity> = prepared
            .entities
            .iter()
            .zip(&spawned)
            .map(|(expanded, &entity)| (expanded.source.id.clone(), entity))
            .collect();
        let entities = Arc::new(entities);
        let context = SceneContext {
            assets: table,
            entities: entities.clone(),
        };
        let landed = self.land(
            world,
            (&prepared, &kinds),
            (&context, &files.buffers()),
            &spawned,
        );
        if let Err(error) = landed {
            for &entity in &spawned {
                let _ = world.try_despawn(entity);
            }
            assets.remove(&mut world.resource_mut::<AssetLibrary>());
            replaced.restore(world);
            return Err(error);
        }
        world.insert_resource(OperationScene::new(self.id.clone(), entities.clone()));
        replaced.retire(world);
        let instance = SceneInstance {
            id: self.id.clone(),
            revision: self.revision,
            world_id: world.id(),
            entities,
            assets,
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
        Ok(instance)
    }

    fn land(
        &self,
        world: &mut World,
        (prepared, kinds): (&PreparedScene, &SceneKinds),
        (context, files): (&SceneContext, &SceneBuffers),
        spawned: &[Entity],
    ) -> Result<(), Error> {
        let cx = SceneCx::new(Some(context), Some(files));
        context.scope(|| {
            prepared
                .entities
                .iter()
                .zip(spawned)
                .try_for_each(|(expanded, &entity)| {
                    kinds.read(world, entity, &expanded.source, &expanded.path, &cx)
                })
        })?;
        let entities = &context.entities;
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
        set_parents(world, &parents).map_err(|error| {
            error
                .coded(ErrorCode::HierarchyInstallFailed)
                .within("$.entities")
        })?;
        propagate_transforms(world);
        world.insert_resource(context.clone());
        let empty = crate::SceneEnvironment::default();
        let environment = prepared.environment.as_ref().unwrap_or(&empty);
        context.scope(|| environment.apply_all(world, "$.environment"))?;
        Ok(())
    }

    fn insert_assets(
        &self,
        world: &mut World,
        prepared: &mut PreparedScene,
    ) -> Result<Handles, Error> {
        let mut handles = Handles::default();
        let mut assets = world
            .get_resource_mut::<AssetLibrary>()
            .ok_or_else(|| Error::new(ErrorCode::MissingAssetLibrary, "$"))?;
        let inserted = (|| {
            insert_each(
                std::mem::take(&mut prepared.meshes),
                &mut handles.meshes,
                "meshes",
                |mesh| assets.insert_mesh(mesh),
            )?;
            insert_each(
                std::mem::take(&mut prepared.textures).into_iter().map(Some),
                &mut handles.textures,
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
                std::mem::take(&mut prepared.materials)
                    .into_iter()
                    .map(Some),
                &mut handles.materials,
                "materials",
                |material| assets.insert_material(material.material(|id| texture_lookup[id])),
            )
        })();
        if let Err(error) = inserted {
            handles.remove(&mut assets);
            return Err(error);
        }
        Ok(handles)
    }

    fn spawn(&self, world: &mut World, prepared: &PreparedScene) -> Vec<Entity> {
        prepared
            .entities
            .iter()
            .zip(&prepared.transforms)
            .map(|(expanded, &transform)| spawn_entity(world, &self.id, expanded, transform))
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

struct Replaced {
    context: Option<SceneContext>,
    scene: Option<OperationScene>,
    sections: Vec<(SettingsEntry, Value)>,
}

impl Replaced {
    fn of(world: &World) -> Self {
        let context = world.get_resource::<SceneContext>().cloned();
        let sections = crate::loading::within(context.as_ref(), || crate::environment::held(world));
        Self {
            context,
            scene: world.get_resource::<OperationScene>().cloned(),
            sections,
        }
    }

    fn retire(self, world: &mut World) {
        for entity in self
            .scene
            .into_iter()
            .flat_map(|scene| scene.installed.into_values())
        {
            let _ = world.try_despawn(entity);
        }
    }

    fn restore(self, world: &mut World) {
        put(world, self.scene);
        put(world, self.context.clone());
        crate::loading::within(self.context.as_ref(), || {
            crate::environment::restore(world, &self.sections);
        });
    }
}

fn put<R: Resource>(world: &mut World, held: Option<R>) {
    match held {
        Some(held) => world.insert_resource(held),
        None => {
            world.remove_resource::<R>();
        }
    }
}

pub(crate) fn spawn_entity(
    world: &mut World,
    scene_id: &str,
    expanded: &ExpandedObject,
    transform: Transform,
) -> Entity {
    let source = &expanded.source;
    world
        .spawn((
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
        ))
        .id()
}
