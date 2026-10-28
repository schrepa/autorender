use engine_settings::ErrorCode;
use std::collections::BTreeMap;
use std::sync::Arc;

use engine_core::{
    AssetError, AssetLibrary, MaterialHandle, Mesh3d, MeshHandle, StreamSet, Streamed,
    StreamedItem, TextureHandle, Transform, WorldCells,
    bevy_ecs::{
        self,
        prelude::{Component, Entity, World},
        world::WorldId,
    },
    propagate_transforms, remove_streamed, set_parents,
};

use crate::{
    SceneDocument, SceneError, SceneFiles,
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

#[derive(Debug)]
#[must_use = "retain the instance and explicitly despawn it when the scene is disposed"]
pub struct SceneInstance {
    id: String,
    revision: u64,
    world_id: WorldId,
    entities: BTreeMap<String, Entity>,
    assets: OwnedAssets,
    material_ids: Vec<String>,
    mesh_ids: Vec<String>,
    texture_ids: Vec<String>,
    origins: Origins,
    streamed: Option<StreamSet>,
}

#[derive(Debug, Default)]
struct Origins {
    meshes: Vec<MeshOrigin>,
    textures: Vec<Option<String>>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MeshOrigin {
    Cube,
    Plane,
    Octahedron,
    Inline {
        vertices: usize,
        indices: usize,
    },
    Buffer {
        uri: String,
    },
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
        }
    }
}

#[derive(Debug, Default)]
struct OwnedAssets {
    meshes: Vec<MeshHandle>,
    materials: Vec<MaterialHandle>,
    textures: Vec<TextureHandle>,
}

impl OwnedAssets {
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

fn insert_each<T, H>(
    items: Vec<T>,
    handles: &mut Vec<H>,
    field: &str,
    mut insert: impl FnMut(T) -> Result<H, AssetError>,
) -> Result<(), SceneError> {
    for (index, item) in items.into_iter().enumerate() {
        let handle = insert(item).map_err(|error| {
            SceneError::new(
                ErrorCode::AssetBudgetExceeded,
                format!("$.{field}[{index}]"),
                error.to_string(),
            )
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

    pub fn texture_origin(&self, handle: TextureHandle) -> Option<(&str, Option<&str>)> {
        let index = self
            .assets
            .textures
            .iter()
            .position(|&texture| texture == handle)?;
        Some((
            &self.texture_ids[index],
            self.origins.textures[index].as_deref(),
        ))
    }

    pub fn entities(&self) -> impl ExactSizeIterator<Item = (&str, Entity)> {
        self.entities
            .iter()
            .map(|(id, entity)| (id.as_str(), *entity))
    }

    pub fn streamed(&self) -> Option<StreamSet> {
        self.streamed
    }

    pub fn belongs_to(&self, world: &World) -> bool {
        self.world_id == world.id()
    }

    pub fn despawn(self, world: &mut World) {
        assert_eq!(self.world_id, world.id(), "scene belongs to another world");
        if let Some(set) = self.streamed {
            remove_streamed(world, set);
        }
        for entity in self.entities.into_values() {
            world.despawn(entity);
        }
        if let Some(mut assets) = world.get_resource_mut::<AssetLibrary>() {
            self.assets.remove(&mut assets);
        }
    }
}

impl SceneDocument {
    pub fn instantiate(&self, world: &mut World) -> Result<SceneInstance, SceneError> {
        self.instantiate_with(world, &SceneFiles::default())
    }

    pub fn instantiate_with(
        &self,
        world: &mut World,
        files: &SceneFiles,
    ) -> Result<SceneInstance, SceneError> {
        let mut prepared = self.prepare(Some(files))?;
        self.check_features(world)?;
        let owned = self.insert_assets(world, &mut prepared)?;
        let lookups = Lookups::new(self, &owned);
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
            return Err(SceneError::new(
                ErrorCode::HierarchyInstallFailed,
                "$.entities",
                error.to_string(),
            ));
        }
        propagate_transforms(world);
        world.insert_resource(self.camera.camera());
        if let Some(environment) = prepared.environment {
            environment.apply_all(world);
        }
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
        Ok(SceneInstance {
            id: self.id.clone(),
            revision: self.revision,
            world_id: world.id(),
            entities,
            assets: owned,
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
                textures: self
                    .textures
                    .iter()
                    .map(|texture| texture.uri.clone())
                    .collect(),
            },
            streamed,
        })
    }

    fn insert_assets(
        &self,
        world: &mut World,
        prepared: &mut PreparedScene,
    ) -> Result<OwnedAssets, SceneError> {
        let mut owned = OwnedAssets::default();
        let mut assets = world.get_resource_mut::<AssetLibrary>().ok_or_else(|| {
            SceneError::new(
                ErrorCode::MissingAssetLibrary,
                "$",
                "install AssetLibrary in the world before instantiating a scene",
            )
        })?;
        let inserted = (|| {
            insert_each(
                std::mem::take(&mut prepared.meshes),
                &mut owned.meshes,
                "meshes",
                |mesh| assets.insert_mesh(mesh),
            )?;
            insert_each(
                std::mem::take(&mut prepared.textures),
                &mut owned.textures,
                "textures",
                |texture| assets.insert_texture(texture),
            )?;
            let texture_lookup: BTreeMap<_, _> = self
                .textures
                .iter()
                .zip(&owned.textures)
                .map(|(source, handle)| (source.id.as_str(), *handle))
                .collect();
            insert_each(
                std::mem::take(&mut prepared.materials),
                &mut owned.materials,
                "materials",
                |material| assets.insert_material(material.material(|id| texture_lookup[id])),
            )
        })();
        if let Err(error) = inserted {
            owned.remove(&mut assets);
            return Err(error);
        }
        Ok(owned)
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

pub(crate) struct Lookups {
    scene_id: String,
    pub(crate) meshes: BTreeMap<String, MeshHandle>,
    pub(crate) materials: BTreeMap<String, MaterialHandle>,
    pub(crate) owned_meshes: Vec<MeshHandle>,
    pub(crate) owned_materials: Vec<MaterialHandle>,
}

impl Lookups {
    fn new(document: &SceneDocument, owned: &OwnedAssets) -> Self {
        Self {
            scene_id: document.id.clone(),
            meshes: document
                .meshes
                .iter()
                .zip(&owned.meshes)
                .map(|(source, handle)| (source.id.clone(), *handle))
                .collect(),
            materials: document
                .materials
                .iter()
                .zip(&owned.materials)
                .map(|(source, handle)| (source.id.clone(), *handle))
                .collect(),
            owned_meshes: owned.meshes.clone(),
            owned_materials: owned.materials.clone(),
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
    component.components.spawn(&mut entity, lookups);
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
                .expect("references stay within an item")]
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
