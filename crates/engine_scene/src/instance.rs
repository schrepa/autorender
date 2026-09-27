use std::collections::BTreeMap;

use engine_core::{
    AssetLibrary, Material, MaterialHandle, Mesh3d, MeshHandle, TextureHandle,
    bevy_ecs::{
        self,
        prelude::{Component, Entity, World},
        world::WorldId,
    },
    propagate_transforms, set_parents,
};

use crate::{SceneDocument, SceneError};

/// Authored identity retained for game setup and development inspection.
///
/// These strings are not read by the renderer or any built-in per-frame system.
/// Multiple instances of the same source may coexist; IDs are scoped to their
/// [`SceneInstance`]. Add game components using tags once after instantiation.
#[derive(Component, Debug)]
pub struct SceneEntity {
    pub scene_id: String,
    pub authored_id: String,
    pub name: String,
    pub tags: Vec<String>,
    /// Expanded authored parent ID; runtime reparenting does not rewrite source.
    pub authored_parent: Option<String>,
    pub prefab_id: Option<String>,
    pub prefab_instance: Option<String>,
    pub prefab_local_id: Option<String>,
}

/// Owns the entities and assets created by one successful scene instantiation.
///
/// Dropping this value does not mutate the world. Call [`Self::despawn`] at an
/// explicit scene disposal boundary. Handles copied into outside entities are
/// weak and become invalid after disposal; old extracted snapshots may retain
/// immutable CPU geometry until they are released. The global camera and
/// environment are not restored during disposal, allowing a new scene to be
/// staged first.
#[derive(Debug)]
#[must_use = "retain the instance and explicitly despawn it when the scene is disposed"]
pub struct SceneInstance {
    id: String,
    revision: u64,
    world_id: WorldId,
    entities: BTreeMap<String, Entity>,
    meshes: Vec<MeshHandle>,
    materials: Vec<MaterialHandle>,
    textures: Vec<TextureHandle>,
    texture_ids: Vec<String>,
}

impl SceneInstance {
    /// Persistent authored scene ID. This is not a unique runtime instance ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Looks up an entity by its authored ID within this instance only.
    ///
    /// The entity may have been removed by gameplay since instantiation; use
    /// the world's checked accessors before reading its components.
    pub fn entity(&self, authored_id: &str) -> Option<Entity> {
        self.entities.get(authored_id).copied()
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Authored texture IDs and their weak CPU handles, in source order.
    /// Readiness inspection belongs at explicit development boundaries.
    pub fn textures(&self) -> impl ExactSizeIterator<Item = (&str, TextureHandle)> {
        self.texture_ids
            .iter()
            .zip(&self.textures)
            .map(|(id, handle)| (id.as_str(), *handle))
    }

    /// Authored IDs in stable lexical order, including entities removed by gameplay.
    /// Check the world's entity accessor before inspecting an entry.
    pub fn entities(&self) -> impl ExactSizeIterator<Item = (&str, Entity)> {
        self.entities
            .iter()
            .map(|(id, entity)| (id.as_str(), *entity))
    }

    /// Whether this instance's runtime entity references belong to this world.
    pub fn belongs_to(&self, world: &World) -> bool {
        self.world_id == world.id()
    }

    /// Removes only owned entities and assets. Already removed items are safe.
    ///
    /// # Panics
    /// Panics before mutation if passed a different world from instantiation.
    pub fn despawn(self, world: &mut World) {
        assert_eq!(self.world_id, world.id(), "scene belongs to another world");
        for entity in self.entities.into_values() {
            world.despawn(entity);
        }
        if let Some(mut assets) = world.get_resource_mut::<AssetLibrary>() {
            remove_assets(&mut assets, &self.meshes, &self.materials, &self.textures);
        }
    }
}

impl SceneDocument {
    /// Validates and creates a complete instance at an exclusive world boundary.
    ///
    /// The caller must install an [`AssetLibrary`]. CPU assets are validated
    /// before insertion; a budget failure rolls back all inserted assets and
    /// leaves existing entities, the camera and the environment unchanged.
    /// Success sets the world's camera to this document's camera and, when the
    /// document has an environment, its atmosphere and effect resources.
    /// There are no GPU uploads here.
    ///
    /// For transactional reload, instantiate a replacement before disposing of
    /// the previous instance. Budgets must allow both instances to coexist.
    pub fn instantiate(&self, world: &mut World) -> Result<SceneInstance, SceneError> {
        let prepared = self.prepare()?;
        let mut meshes = Vec::with_capacity(prepared.meshes.len());
        let mut materials = Vec::with_capacity(prepared.materials.len());
        let mut textures = Vec::with_capacity(prepared.textures.len());
        {
            let mut assets = world.get_resource_mut::<AssetLibrary>().ok_or_else(|| {
                SceneError::new(
                    "missing_asset_library",
                    "$",
                    "install AssetLibrary in the world before instantiating a scene",
                )
            })?;
            for (index, mesh) in prepared.meshes.into_iter().enumerate() {
                match assets.insert_mesh(mesh) {
                    Ok(handle) => meshes.push(handle),
                    Err(error) => {
                        remove_assets(&mut assets, &meshes, &materials, &textures);
                        return Err(SceneError::new(
                            "asset_budget_exceeded",
                            format!("$.meshes[{index}]"),
                            error.to_string(),
                        ));
                    }
                }
            }
            for (index, texture) in prepared.textures.into_iter().enumerate() {
                match assets.insert_texture(texture) {
                    Ok(handle) => textures.push(handle),
                    Err(error) => {
                        remove_assets(&mut assets, &meshes, &materials, &textures);
                        return Err(SceneError::new(
                            "asset_budget_exceeded",
                            format!("$.textures[{index}]"),
                            error.to_string(),
                        ));
                    }
                }
            }
            let texture_lookup: BTreeMap<_, _> = self
                .textures
                .iter()
                .zip(&textures)
                .map(|(source, handle)| (source.id.as_str(), *handle))
                .collect();
            for (index, material) in prepared.materials.into_iter().enumerate() {
                let material = match &self.materials[index].texture {
                    Some(id) => Material::textured(material.color(), texture_lookup[id.as_str()])
                        .and_then(|textured| textured.with_surface(material.surface()))
                        .expect("prepared opaque material remains valid with a texture"),
                    None => material,
                };
                match assets.insert_material(material) {
                    Ok(handle) => materials.push(handle),
                    Err(error) => {
                        remove_assets(&mut assets, &meshes, &materials, &textures);
                        return Err(SceneError::new(
                            "asset_budget_exceeded",
                            format!("$.materials[{index}]"),
                            error.to_string(),
                        ));
                    }
                }
            }
        }
        let mesh_lookup: BTreeMap<_, _> = self
            .meshes
            .iter()
            .zip(&meshes)
            .map(|(source, handle)| (source.id.as_str(), *handle))
            .collect();
        let material_lookup: BTreeMap<_, _> = self
            .materials
            .iter()
            .zip(&materials)
            .map(|(source, handle)| (source.id.as_str(), *handle))
            .collect();
        let mut entities = BTreeMap::new();
        for ((expanded, transform), collider) in prepared
            .entities
            .iter()
            .zip(prepared.transforms)
            .zip(prepared.colliders)
        {
            let source = &expanded.source;
            let mut entity = world.spawn((
                transform,
                SceneEntity {
                    scene_id: self.id.clone(),
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
                    mesh: mesh_lookup[mesh.as_str()],
                    material: material_lookup[material.as_str()],
                });
            }
            if let Some(collider) = collider {
                entity.insert(collider);
            }
            entities.insert(source.id.clone(), entity.id());
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
            let mut assets = world.resource_mut::<AssetLibrary>();
            remove_assets(&mut assets, &meshes, &materials, &textures);
            return Err(SceneError::new(
                "hierarchy_install_failed",
                "$.entities",
                error.to_string(),
            ));
        }
        propagate_transforms(world);
        world.insert_resource(self.camera.camera());
        if let Some(environment) = prepared.environment {
            environment.apply_all(world);
        }
        Ok(SceneInstance {
            id: self.id.clone(),
            revision: self.revision,
            world_id: world.id(),
            entities,
            meshes,
            materials,
            textures,
            texture_ids: self
                .textures
                .iter()
                .map(|texture| texture.id.clone())
                .collect(),
        })
    }
}

fn remove_assets(
    assets: &mut AssetLibrary,
    meshes: &[MeshHandle],
    materials: &[MaterialHandle],
    textures: &[TextureHandle],
) {
    for &handle in meshes {
        assets.remove_mesh(handle);
    }
    for &handle in materials {
        assets.remove_material(handle);
    }
    for &handle in textures {
        assets.remove_texture(handle);
    }
}
