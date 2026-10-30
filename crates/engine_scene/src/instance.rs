use engine_settings::ErrorCode;
use std::collections::BTreeMap;
use std::sync::Arc;

use engine_core::{
    AssetLibrary, Error, MaterialHandle, Mesh3d, MeshHandle, ProvidedAssets, TextureHandle,
    Transform,
    bevy_ecs::{
        self,
        prelude::{Component, Entity, World},
        world::WorldId,
    },
    propagate_transforms, set_parents,
};
use engine_world_streaming::{StreamSet, Streamed, StreamedItem, WorldCells, remove_streamed};

use crate::{
    SceneDocument, SceneFiles,
    hierarchy::ExpandedObject,
    validation::{PreparedEntity, PreparedScene, PreparedStreamed},
};

#[derive(Component, Debug)]
pub struct SceneEntity {
    pub scene_id: String,
    pub authored_id: String,
    pub name: String,
    pub tags: Vec<String>,
    pub authored_parent: Option<String>,
    pub prefab_id: Option<String>,
    pub prefab_instance: Option<String>,
    pub prefab_local_id: Option<String>,
}

pub fn authored_id(world: &World, entity: Entity) -> Option<engine_core::AuthoredId> {
    world
        .get::<SceneEntity>(entity)
        .map(|metadata| engine_core::AuthoredId {
            scene: metadata.scene_id.clone(),
            id: metadata.authored_id.clone(),
            name: metadata.name.clone(),
        })
}

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
    streamed: Option<StreamSet>,
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

    pub fn entity_count(&self) -> usize {
        self.entities.len()
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

    pub fn operation_scene(&self) -> engine_core::OperationScene {
        engine_core::OperationScene {
            id: self.id.clone(),
            entities: self.entities.clone(),
            streamed: self.streamed.is_some(),
            authored: authored_id,
        }
    }

    pub fn streamed(&self) -> Option<StreamSet> {
        self.streamed
    }

    pub fn belongs_to(&self, world: &World) -> bool {
        self.world_id == world.id()
    }

    pub fn despawn(self, world: &mut World) {
        assert_eq!(self.world_id, world.id(), "scene_world_matches");
        if world
            .get_resource::<engine_core::OperationScene>()
            .is_some_and(|scene| Arc::ptr_eq(&scene.entities, &self.entities))
        {
            world.remove_resource::<engine_core::OperationScene>();
        }
        if let Some(set) = self.streamed {
            remove_streamed(world, set);
        }
        for &entity in self.entities.values() {
            world.despawn(entity);
        }
        if let Some(mut assets) = world.get_resource_mut::<AssetLibrary>() {
            self.owned.remove(&mut assets);
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
        let mut prepared = self.prepare(Some(files))?;
        self.check_bricks(world)?;
        let (assets, owned) = self.insert_assets(world, &mut prepared)?;
        let lookups = Lookups::new(self, &assets);
        let spawned = self.spawn(world, &prepared, &lookups);
        let entities: BTreeMap<String, Entity> = prepared
            .entities
            .iter()
            .zip(&spawned)
            .map(|(expanded, &entity)| (expanded.source.id.clone(), entity))
            .collect();
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
        if let Some(environment) = &prepared.environment {
            environment.apply_all(world, "$.environment")?;
        }
        world.insert_resource(crate::SceneViews::of(self));
        let streamed = (!prepared.streamed.items.is_empty()).then(|| {
            let scene = Arc::new(StreamedScene {
                lookups,
                prepared: std::mem::take(&mut prepared.streamed),
            });
            let items: Vec<StreamedItem> = scene
                .prepared
                .items
                .iter()
                .enumerate()
                .map(|(item, prepared)| StreamedItem {
                    id: Some(Arc::from(format!(
                        "{}/{}",
                        self.id, scene.prepared.entities[prepared.entities[0]].source.id
                    ))),
                    position: prepared.position,
                    reach: prepared.reach,
                    content: Arc::new(StreamedTree {
                        scene: scene.clone(),
                        item,
                    }),
                })
                .collect();
            world
                .get_resource_or_insert_with(WorldCells::default)
                .add(items)
        });
        let instance = SceneInstance {
            id: self.id.clone(),
            revision: self.revision,
            world_id: world.id(),
            entities: Arc::new(entities),
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
            streamed,
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

    fn spawn(&self, world: &mut World, prepared: &PreparedScene, lookups: &Lookups) -> Vec<Entity> {
        prepared
            .entities
            .iter()
            .zip(&prepared.transforms)
            .zip(&prepared.components)
            .map(|((expanded, &transform), component)| {
                spawn_entity(world, lookups, expanded, transform, component)
            })
            .collect()
    }
}

pub struct Lookups {
    scene_id: String,
    pub(crate) meshes: BTreeMap<String, MeshHandle>,
    pub(crate) materials: BTreeMap<String, MaterialHandle>,
    pub owned_meshes: Vec<MeshHandle>,
    pub owned_materials: Vec<MaterialHandle>,
}

impl Lookups {
    fn new(document: &SceneDocument, assets: &Handles) -> Self {
        Self {
            scene_id: document.id.clone(),
            meshes: document
                .meshes
                .iter()
                .zip(&assets.meshes)
                .map(|(source, handle)| (source.id.clone(), *handle))
                .collect(),
            materials: document
                .materials
                .iter()
                .zip(&assets.materials)
                .map(|(source, handle)| (source.id.clone(), *handle))
                .collect(),
            owned_meshes: assets.meshes.clone(),
            owned_materials: assets.materials.clone(),
        }
    }
}

fn spawn_entity(
    world: &mut World,
    lookups: &Lookups,
    expanded: &ExpandedObject,
    transform: Transform,
    component: &PreparedEntity,
) -> Entity {
    let source = &expanded.source;
    let mut entity = world.spawn((
        transform,
        SceneEntity {
            scene_id: lookups.scene_id.clone(),
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
            mesh: lookups.meshes[mesh.as_str()],
            material: lookups.materials[material.as_str()],
        });
    }
    for prepared in &component.components {
        prepared.spawn(&mut entity, lookups);
    }
    entity.id()
}

struct StreamedScene {
    lookups: Lookups,
    prepared: PreparedStreamed,
}

struct StreamedTree {
    scene: Arc<StreamedScene>,
    item: usize,
}

impl Streamed for StreamedTree {
    fn entities(&self) -> u32 {
        self.scene.prepared.items[self.item].entities.len() as u32
    }

    fn spawn(&self, world: &mut World, spawned: &mut Vec<Entity>) {
        let scene = &*self.scene;
        let prepared = &scene.prepared;
        let item = &prepared.items[self.item];
        let mine: Vec<Entity> = item
            .entities
            .iter()
            .map(|&index| {
                spawn_entity(
                    world,
                    &scene.lookups,
                    &prepared.entities[index],
                    prepared.transforms[index],
                    &prepared.components[index],
                )
            })
            .collect();
        let entity_of = |index: usize| {
            mine[item
                .entities
                .iter()
                .position(|&own| own == index)
                .expect("reference_within_item")]
        };
        let parents: Vec<_> = item
            .entities
            .iter()
            .zip(&mine)
            .filter_map(|(&index, &entity)| {
                prepared.parents[index].map(|parent| (entity, Some(entity_of(parent))))
            })
            .collect();
        if set_parents(world, &parents).is_err() {
            for entity in mine {
                let _ = world.try_despawn(entity);
            }
            return;
        }
        spawned.extend(mine);
    }
}
