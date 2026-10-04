use std::collections::BTreeMap;

use engine_core::{
    AnimationPlayer, AssetError, AssetLibrary, ClipHandle, InterpolateTransform, MaterialHandle,
    Mesh3d, MeshHandle, Skin, TextureHandle,
    bevy_ecs::{
        self,
        prelude::{Component, Entity, World},
        world::WorldId,
    },
    propagate_transforms, set_parents,
};

use crate::{SceneDocument, SceneError, SceneFiles, validation::PreparedScene};

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
    assets: OwnedAssets,
    material_ids: Vec<String>,
    texture_ids: Vec<String>,
    clip_ids: Vec<String>,
}

/// The assets one instantiation inserted, in source order, removed together.
#[derive(Debug, Default)]
struct OwnedAssets {
    meshes: Vec<MeshHandle>,
    materials: Vec<MaterialHandle>,
    textures: Vec<TextureHandle>,
    clips: Vec<ClipHandle>,
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
        for &clip in &self.clips {
            assets.remove_clip(clip);
        }
    }
}

/// Inserts `items` with `insert`, pushing each handle onto `handles`; the
/// first refusal is a budget error at `$.{field}[index]`.
fn insert_each<T, H>(
    items: Vec<T>,
    handles: &mut Vec<H>,
    field: &str,
    mut insert: impl FnMut(T) -> Result<H, AssetError>,
) -> Result<(), SceneError> {
    for (index, item) in items.into_iter().enumerate() {
        let handle = insert(item).map_err(|error| {
            SceneError::new(
                "asset_budget_exceeded",
                format!("$.{field}[{index}]"),
                error.to_string(),
            )
        })?;
        handles.push(handle);
    }
    Ok(())
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
            .zip(&self.assets.textures)
            .map(|(id, handle)| (id.as_str(), *handle))
    }

    /// Authored material IDs and their weak handles, in source order.
    pub fn materials(&self) -> impl ExactSizeIterator<Item = (&str, MaterialHandle)> {
        self.material_ids
            .iter()
            .zip(&self.assets.materials)
            .map(|(id, handle)| (id.as_str(), *handle))
    }

    /// The clip of animation `id` (decision 035), for switching what an
    /// [`AnimationPlayer`] plays. Handles are weak, like texture handles.
    pub fn clip(&self, id: &str) -> Option<ClipHandle> {
        self.clip_ids
            .iter()
            .position(|clip| clip == id)
            .map(|index| self.assets.clips[index])
    }

    /// Authored animation IDs and their clips, in source order.
    pub fn clips(&self) -> impl ExactSizeIterator<Item = (&str, ClipHandle)> {
        self.clip_ids
            .iter()
            .zip(&self.assets.clips)
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
            self.assets.remove(&mut assets);
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
        self.instantiate_with(world, &SceneFiles::default())
    }

    /// [`Self::instantiate`] with the contents of the files the document
    /// references (decision 030); a missing or invalid one fails first.
    pub fn instantiate_with(
        &self,
        world: &mut World,
        files: &SceneFiles,
    ) -> Result<SceneInstance, SceneError> {
        let mut prepared = self.prepare(Some(files))?;
        let owned = self.insert_assets(world, &mut prepared)?;
        let spawned = self.spawn(world, &prepared, &owned);
        attach_rigs(world, &prepared, &owned, &spawned);
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
            assets: owned,
            material_ids: self
                .materials
                .iter()
                .map(|material| material.id.clone())
                .collect(),
            texture_ids: self
                .textures
                .iter()
                .map(|texture| texture.id.clone())
                .collect(),
            clip_ids: self
                .animations
                .iter()
                .map(|animation| animation.id.clone())
                .collect(),
        })
    }

    /// Inserts `prepared`'s meshes, textures, materials and clips: a refused
    /// one leaves none of this document's assets resident.
    fn insert_assets(
        &self,
        world: &mut World,
        prepared: &mut PreparedScene,
    ) -> Result<OwnedAssets, SceneError> {
        let mut owned = OwnedAssets::default();
        let mut assets = world.get_resource_mut::<AssetLibrary>().ok_or_else(|| {
            SceneError::new(
                "missing_asset_library",
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
            )?;
            insert_each(
                std::mem::take(&mut prepared.clips),
                &mut owned.clips,
                "animations",
                |clip| assets.insert_clip(clip),
            )
        })();
        if let Err(error) = inserted {
            owned.remove(&mut assets);
            return Err(error);
        }
        Ok(owned)
    }

    /// Spawns each expanded entity with its transform, identity, mesh and
    /// components; returns them in expanded order.
    fn spawn(
        &self,
        world: &mut World,
        prepared: &PreparedScene,
        owned: &OwnedAssets,
    ) -> Vec<Entity> {
        let mesh_lookup: BTreeMap<_, _> = self
            .meshes
            .iter()
            .zip(&owned.meshes)
            .map(|(source, handle)| (source.id.as_str(), *handle))
            .collect();
        let material_lookup: BTreeMap<_, _> = self
            .materials
            .iter()
            .zip(&owned.materials)
            .map(|(source, handle)| (source.id.as_str(), *handle))
            .collect();
        let mut spawned = Vec::with_capacity(prepared.entities.len());
        for ((expanded, &transform), component) in prepared
            .entities
            .iter()
            .zip(&prepared.transforms)
            .zip(&prepared.components)
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
            if let Some(collider) = component.collider {
                entity.insert(collider);
            }
            if let Some(light) = component.light {
                entity.insert(light);
            }
            if let Some(emitter) = component.emitter {
                entity.insert(emitter);
            }
            if let Some(tint) = component.tint {
                entity.insert(tint);
            }
            if let Some(decal) = &component.decal {
                entity.insert(decal.runtime(owned.materials[decal.material]));
            }
            if let Some(probe) = component.probe {
                entity.insert(probe);
            }
            spawned.push(entity.id());
        }
        spawned
    }
}

/// Gives the `spawned` entities their animation players and skins, which
/// name other spawned entities; animated targets present smoothly between
/// fixed ticks.
fn attach_rigs(
    world: &mut World,
    prepared: &PreparedScene,
    owned: &OwnedAssets,
    spawned: &[Entity],
) {
    for (component, &entity) in prepared.components.iter().zip(spawned) {
        if let Some(player) = &component.player {
            for &target in &player.targets {
                world
                    .entity_mut(spawned[target])
                    .insert(InterpolateTransform);
            }
            world.entity_mut(entity).insert(AnimationPlayer {
                clip: owned.clips[player.clip],
                targets: player
                    .targets
                    .iter()
                    .map(|&target| Some(spawned[target]))
                    .collect(),
                time: player.time,
                speed: player.speed,
                looping: player.looping,
            });
        }
    }
    for (component, &entity) in prepared.components.iter().zip(spawned) {
        if let Some(skin) = &component.skin {
            world.entity_mut(entity).insert(Skin::new(
                skin.joints.iter().map(|&joint| spawned[joint]).collect(),
                prepared.skin_binds[skin.skin].clone(),
            ));
        }
    }
}
