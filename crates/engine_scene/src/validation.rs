use engine_settings::ErrorCode;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use engine_core::{
    AnimationClip, LocalLight, Mesh, ParticleEmitter, ReflectionProbe, Texture, Tint, Transform,
    glam::{Mat4, Vec3},
};

use crate::error::broken_rule;
use crate::{
    MeshSource, SceneDocument, SceneError, SceneFile, SceneFileKind, SceneFiles,
    animation::{PreparedPlayer, PreparedSkin},
    decal::PreparedDecal,
    hierarchy::{ExpandedObject, ExpandedScene},
    material::PreparedMaterial,
};

pub(crate) struct PreparedScene {
    /// Validated.
    pub environment: Option<crate::SceneEnvironment>,
    /// The components of each expanded entity, parallel to `entities`.
    pub components: Vec<PreparedEntity>,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<PreparedMaterial>,
    pub textures: Vec<Texture>,
    pub transforms: Vec<Transform>,
    pub entities: Vec<ExpandedObject>,
    pub clips: Vec<AnimationClip>,
    pub skin_binds: Vec<Arc<[Mat4]>>,
    /// Streamed entities.
    pub streamed: PreparedStreamed,
}

/// A document's streamed entities, expanded: their
/// transforms, components and parents (indices into this list), and the
/// items they form.
#[derive(Default)]
pub(crate) struct PreparedStreamed {
    pub entities: Vec<ExpandedObject>,
    pub transforms: Vec<Transform>,
    pub components: Vec<PreparedEntity>,
    pub parents: Vec<Option<usize>>,
    pub items: Vec<PreparedItem>,
}

/// A streamed root with its descendants: their indices,
/// the root first and the rest in expanded order; where the root stands;
/// and the meters its content reaches from there.
pub(crate) struct PreparedItem {
    pub entities: Vec<usize>,
    pub position: Vec3,
    pub reach: f32,
}

/// An expanded entity's validated components besides its transform, mesh
/// and material.
pub(crate) struct PreparedEntity {
    pub collider: Option<engine_collision::Collider>,
    pub light: Option<LocalLight>,
    pub emitter: Option<ParticleEmitter>,
    pub tint: Option<Tint>,
    pub decal: Option<PreparedDecal>,
    pub probe: Option<ReflectionProbe>,
    pub player: Option<PreparedPlayer>,
    pub skin: Option<PreparedSkin>,
    /// Prepared with the scene's files.
    pub terrain: Option<crate::terrain::PreparedTerrain>,
}

/// Resolves an ID a clip target or skin joint names, from an entity's
/// placement first, to an expanded entity index.
type Resolve<'a> = dyn Fn(&ExpandedObject, &str) -> Option<usize> + 'a;

/// A document's animation and skin IDs, and each skin's inverse bind
/// matrices.
struct Rigs<'a> {
    animation_ids: BTreeSet<&'a str>,
    skin_ids: BTreeSet<&'a str>,
    skin_binds: Vec<Arc<[Mat4]>>,
}

/// A document's CPU assets, in source order.
struct CpuAssets {
    textures: Vec<Texture>,
    meshes: Vec<Mesh>,
    materials: Vec<PreparedMaterial>,
}

impl SceneDocument {
    /// Validates a document assembled through Rust or an authoring editor.
    ///
    /// Validation is independent of a world and may allocate temporary CPU
    /// geometry. Asset residency budgets are checked during instantiation.
    /// Referenced files are checked only by [`Self::validate_with`].
    pub fn validate(&self) -> Result<(), SceneError> {
        self.prepare(None).map(|_| ())
    }

    /// Validates the document and the contents of the files it references.
    pub fn validate_with(&self, files: &SceneFiles) -> Result<(), SceneError> {
        self.prepare(Some(files)).map(|_| ())
    }

    /// The files this document references, each once, in
    /// authored order: the host supplies them as [`SceneFiles`].
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
                MeshSource::Virtual { uri, .. } => add(uri, SceneFileKind::VirtualMesh),
                MeshSource::Cube { .. }
                | MeshSource::Plane { .. }
                | MeshSource::Octahedron { .. }
                | MeshSource::Inline { .. } => {}
            }
        }
        for animation in &self.animations {
            if let Some(uri) = &animation.uri {
                add(uri, SceneFileKind::Buffer);
            }
        }
        // Terrains: heightmaps as buffers, and terrain files.
        let entities = self
            .entities
            .iter()
            .chain(self.prefabs.iter().flat_map(|prefab| &prefab.entities));
        for entity in entities {
            if let Some(terrain) = &entity.terrain {
                for (uri, file) in terrain.files() {
                    add(
                        uri,
                        if file {
                            SceneFileKind::Terrain
                        } else {
                            SceneFileKind::Buffer
                        },
                    );
                }
            }
        }
        files
    }

    /// `files`: `None` validates structure only, reading referenced files as
    /// placeholders.
    pub(crate) fn prepare(&self, files: Option<&SceneFiles>) -> Result<PreparedScene, SceneError> {
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
        let rigs = self.prepare_rigs()?;
        let entity_indices: HashMap<&str, usize> = expanded
            .entities
            .iter()
            .enumerate()
            .map(|(index, entity)| (entity.source.id.as_str(), index))
            .collect();
        // Targets and joints name local IDs of the entity's placement first,
        // then scene IDs.
        let resolve = |entity: &ExpandedObject, id: &str| {
            entity
                .prefab_instance
                .as_ref()
                .and_then(|instance| entity_indices.get(format!("{instance}.{id}").as_str()))
                .or_else(|| entity_indices.get(id))
                .copied()
        };
        let mut components = expanded
            .entities
            .iter()
            .map(|entity| self.prepare_entity(entity, &resolve, &rigs))
            .collect::<Result<Vec<_>, _>>()?;
        // Terrains read their files.
        for (entity, component) in expanded.entities.iter().zip(&mut components) {
            component.terrain = entity
                .source
                .terrain
                .as_ref()
                .map(|terrain| {
                    terrain.prepare(
                        &format!("{}.terrain", entity.path),
                        files,
                        (&self.materials, &self.meshes),
                    )
                })
                .transpose()?;
        }

        let streamed_components = self.prepare_streamed(&streamed, &rigs)?;

        let CpuAssets {
            textures,
            meshes,
            materials,
        } = self.prepare_assets(files)?;
        self.check_skin_joints(&expanded.entities, &components, &meshes)?;
        self.check_skin_joints(&streamed.entities, &streamed_components, &meshes)?;
        let streamed = self.streamed_items(streamed, streamed_components, &meshes);
        let clips = self
            .animations
            .iter()
            .enumerate()
            .map(|(index, animation)| animation.clip(&format!("$.animations[{index}]"), files))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(environment) = &self.environment {
            environment.validate("$.environment")?;
        }
        Ok(PreparedScene {
            environment: self.environment,
            components,
            meshes,
            materials,
            textures,
            transforms: expanded.transforms,
            entities: expanded.entities,
            clips,
            skin_binds: rigs.skin_binds,
            streamed,
        })
    }

    /// The components of the streamed entities, whose
    /// animation targets and skin joints resolve within their item. No
    /// terrains: they stream their own tiles.
    fn prepare_streamed(
        &self,
        streamed: &ExpandedScene,
        rigs: &Rigs<'_>,
    ) -> Result<Vec<PreparedEntity>, SceneError> {
        let indices: HashMap<&str, usize> = streamed
            .entities
            .iter()
            .enumerate()
            .map(|(index, entity)| (entity.source.id.as_str(), index))
            .collect();
        let roots = item_roots(&streamed.entities, &indices);
        let resolve = |entity: &ExpandedObject, id: &str| {
            let own = indices[entity.source.id.as_str()];
            entity
                .prefab_instance
                .as_ref()
                .and_then(|instance| indices.get(format!("{instance}.{id}").as_str()))
                .or_else(|| indices.get(id))
                .copied()
                .filter(|&target| roots[target] == roots[own])
        };
        streamed
            .entities
            .iter()
            .map(|entity| {
                if entity.source.terrain.is_some() {
                    return Err(SceneError::new(
                        ErrorCode::InvalidStreamed,
                        format!("{}.terrain", entity.path),
                        "terrains stream their own tiles; author them in entities",
                    ));
                }
                self.prepare_entity(entity, &resolve, rigs)
            })
            .collect()
    }

    /// The streamed entities' items: each root with its
    /// descendants, standing at the root's position and reaching as far as
    /// their meshes, lights, emitters, decals, probes and splines do.
    fn streamed_items(
        &self,
        streamed: ExpandedScene,
        components: Vec<PreparedEntity>,
        meshes: &[Mesh],
    ) -> PreparedStreamed {
        let indices: HashMap<&str, usize> = streamed
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
        // World matrices, parents first.
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
                    worlds[parent].expect("parent composed before child") * local
                }));
            }
        }
        let mesh_indices: HashMap<&str, usize> = self
            .meshes
            .iter()
            .enumerate()
            .map(|(index, mesh)| (mesh.id.as_str(), index))
            .collect();
        let mut items: Vec<PreparedItem> = Vec::new();
        let mut item_of = HashMap::new();
        for (index, &root) in roots.iter().enumerate() {
            let item = *item_of.entry(root).or_insert_with(|| {
                items.push(PreparedItem {
                    entities: Vec::new(),
                    position: worlds[root]
                        .expect("every world matrix is composed")
                        .w_axis
                        .truncate(),
                    reach: 0.0,
                });
                items.len() - 1
            });
            items[item].entities.push(index);
            let position = items[item].position;
            let world = worlds[index].expect("every world matrix is composed");
            let from = world.w_axis.truncate().distance(position);
            let box_reach = |min: Vec3, max: Vec3| {
                (0..8)
                    .map(|corner| {
                        let pick = |bit: u32, low: f32, high: f32| {
                            if corner & (1 << bit) == 0 { low } else { high }
                        };
                        let point = Vec3::new(
                            pick(0, min.x, max.x),
                            pick(1, min.y, max.y),
                            pick(2, min.z, max.z),
                        );
                        world.transform_point3(point).distance(position)
                    })
                    .fold(0.0, engine_core::math::max)
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
                extend(box_reach(bounds.min(), bounds.max()));
            }
            if let Some(light) = component.light {
                extend(from + light.range);
            }
            if let Some(emitter) = component.emitter {
                extend(
                    from + emitter.offset.length()
                        + emitter.radius
                        + emitter.speed * emitter.lifetime
                        + 0.5 * emitter.acceleration.length() * emitter.lifetime * emitter.lifetime,
                );
            }
            if component.decal.is_some() {
                extend(box_reach(Vec3::splat(-0.5), Vec3::splat(0.5)));
            }
            if let Some(probe) = component.probe {
                extend(from + probe.extent.length());
            }
            if let Some(spline) = &source.spline {
                let size = mesh_indices.get(spline.mesh.as_str()).map_or(0.0, |&mesh| {
                    let bounds = meshes[mesh].bounds();
                    (bounds.max() - bounds.min()).length()
                });
                for point in &spline.points {
                    extend(
                        world
                            .transform_point3(Vec3::from_array(*point))
                            .distance(position)
                            + size,
                    );
                }
            }
            if reach.is_finite() {
                items[item].reach = engine_core::math::max(items[item].reach, reach);
            }
        }
        // Each item's root first.
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

    /// Animation and skin sources: their unique IDs,
    /// names, targets and joints, the clips and skins prefab definitions
    /// name, and each skin's inverse bind matrices.
    fn prepare_rigs(&self) -> Result<Rigs<'_>, SceneError> {
        let animation_ids = unique_ids(
            self.animations
                .iter()
                .map(|animation| animation.id.as_str()),
            "$.animations",
        )?;
        let skin_ids = unique_ids(self.skins.iter().map(|skin| skin.id.as_str()), "$.skins")?;
        for (index, animation) in self.animations.iter().enumerate() {
            if animation.name.len() > 128 || animation.name.chars().any(char::is_control) {
                return Err(SceneError::new(
                    ErrorCode::InvalidName,
                    format!("$.animations[{index}].name"),
                    "names must be at most 128 UTF-8 bytes without control characters",
                ));
            }
            for (target, id) in animation.targets.iter().enumerate() {
                identifier(id, &format!("$.animations[{index}].targets[{target}]"))?;
            }
        }
        let skin_binds = self
            .skins
            .iter()
            .enumerate()
            .map(|(index, skin)| {
                for (joint, id) in skin.joints.iter().enumerate() {
                    identifier(id, &format!("$.skins[{index}].joints[{joint}]"))?;
                }
                skin.inverse_binds(&format!("$.skins[{index}]"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Prefab definitions reference clips and skins even when unplaced.
        for (prefab, definition) in self.prefabs.iter().enumerate() {
            for (index, entity) in definition.entities.iter().enumerate() {
                let path = format!("$.prefabs[{prefab}].entities[{index}]");
                if let Some(player) = &entity.animation {
                    reference(
                        &player.clip,
                        &animation_ids,
                        &format!("{path}.animation.clip"),
                    )?;
                }
                if let Some(skin) = &entity.skin {
                    reference(skin, &skin_ids, &format!("{path}.skin"))?;
                }
            }
        }
        Ok(Rigs {
            animation_ids,
            skin_ids,
            skin_binds,
        })
    }

    /// Every CPU texture, mesh and material, built before a world resource
    /// is borrowed: an invalid late asset cannot leave earlier ones
    /// resident. `files` as for [`Self::prepare`].
    fn prepare_assets(&self, files: Option<&SceneFiles>) -> Result<CpuAssets, SceneError> {
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

    /// Refuses a skin with fewer joints than its entity's mesh names.
    fn check_skin_joints(
        &self,
        entities: &[ExpandedObject],
        components: &[PreparedEntity],
        meshes: &[Mesh],
    ) -> Result<(), SceneError> {
        for (entity, component) in entities.iter().zip(components) {
            let (Some(skin), Some(mesh)) = (&component.skin, &entity.source.mesh) else {
                continue;
            };
            let mesh = &meshes[self
                .meshes
                .iter()
                .position(|source| source.id == *mesh)
                .expect("referenced mesh exists")];
            if mesh.joint_count() as usize > skin.joints.len() {
                return Err(SceneError::new(
                    ErrorCode::InvalidSkin,
                    format!("{}.skin", entity.path),
                    format!(
                        "the mesh's vertices name {} joints but skin '{}' has {}",
                        mesh.joint_count(),
                        self.skins[skin.skin].id,
                        skin.joints.len()
                    ),
                ));
            }
        }
        Ok(())
    }

    /// The components of expanded entity `entity`, validated, with clip
    /// targets and skin joints resolved by `resolve`.
    fn prepare_entity(
        &self,
        entity: &ExpandedObject,
        resolve: &Resolve<'_>,
        rigs: &Rigs<'_>,
    ) -> Result<PreparedEntity, SceneError> {
        let (source, path) = (&entity.source, &entity.path);
        // The components' rules are checked with the entity's.
        Ok(PreparedEntity {
            player: self.prepare_player(entity, resolve, &rigs.animation_ids)?,
            skin: self.prepare_skin(entity, resolve, &rigs.skin_ids)?,
            collider: source
                .collider
                .map(|collider| collider.runtime(&format!("{path}.collider")))
                .transpose()?,
            light: source.light.map(|light| light.runtime()),
            emitter: source.particles.map(|emitter| emitter.runtime()),
            tint: source.tint.map(Tint),
            decal: source
                .decal
                .as_ref()
                .map(|decal| decal.prepare(&self.materials)),
            probe: source.reflection_probe.map(|probe| probe.runtime()),
            terrain: None,
        })
    }

    /// The animation player of `entity`, its targets resolved by `resolve`.
    fn prepare_player(
        &self,
        entity: &ExpandedObject,
        resolve: &Resolve<'_>,
        animation_ids: &BTreeSet<&str>,
    ) -> Result<Option<PreparedPlayer>, SceneError> {
        let Some(player) = &entity.source.animation else {
            return Ok(None);
        };
        let path = format!("{}.animation", entity.path);
        reference(&player.clip, animation_ids, &format!("{path}.clip"))?;
        if !player.speed.is_finite() || player.speed.abs() > 1000.0 {
            return Err(SceneError::new(
                ErrorCode::InvalidAnimation,
                format!("{path}.speed"),
                "speed must be finite and within ±1000",
            ));
        }
        if !player.time.is_finite() || player.time < 0.0 {
            return Err(SceneError::new(
                ErrorCode::InvalidAnimation,
                format!("{path}.time"),
                "time must be finite and not negative",
            ));
        }
        let clip = self
            .animations
            .iter()
            .position(|animation| animation.id == player.clip)
            .expect("referenced clip exists");
        let targets = self.animations[clip]
            .targets
            .iter()
            .map(|target| {
                resolve(entity, target).ok_or_else(|| {
                    SceneError::new(
                        ErrorCode::UnknownReference,
                        format!("{path}.clip"),
                        format!(
                            "target '{target}' of animation '{}' matches no entity of this placement or scene",
                            player.clip
                        ),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(PreparedPlayer {
            clip,
            targets,
            speed: player.speed,
            looping: player.looping,
            time: player.time,
        }))
    }

    /// The skin of `entity`, its joints resolved by `resolve`.
    fn prepare_skin(
        &self,
        entity: &ExpandedObject,
        resolve: &Resolve<'_>,
        skin_ids: &BTreeSet<&str>,
    ) -> Result<Option<PreparedSkin>, SceneError> {
        let Some(id) = &entity.source.skin else {
            return Ok(None);
        };
        let path = format!("{}.skin", entity.path);
        reference(id, skin_ids, &path)?;
        if entity.source.mesh.is_none() {
            return Err(SceneError::new(
                ErrorCode::InvalidSkin,
                path,
                "a skin deforms its entity's mesh; give the entity a mesh and material",
            ));
        }
        let skin = self
            .skins
            .iter()
            .position(|skin| skin.id == *id)
            .expect("referenced skin exists");
        let joints = self.skins[skin]
            .joints
            .iter()
            .map(|joint| {
                resolve(entity, joint).ok_or_else(|| {
                    SceneError::new(
                        ErrorCode::UnknownReference,
                        path.clone(),
                        format!(
                            "joint '{joint}' of skin '{id}' matches no entity of this placement or scene"
                        ),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(PreparedSkin { skin, joints }))
    }

    fn validate_camera(&self) -> Result<(), SceneError> {
        self.camera
            .validate()
            .map_err(|invalid| broken_rule(ErrorCode::InvalidCamera, "$.camera", invalid))
    }
}

/// The root of each expanded entity's tree, by index; `indices` maps IDs
/// to indices. The graph is validated: parents exist and form no cycle.
fn item_roots(entities: &[ExpandedObject], indices: &HashMap<&str, usize>) -> Vec<usize> {
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

pub(crate) fn identifier(id: &str, path: &str) -> Result<(), SceneError> {
    if id.is_empty()
        || id.len() > 64
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(SceneError::new(
            ErrorCode::InvalidId,
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
                ErrorCode::DuplicateId,
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
            ErrorCode::UnknownReference,
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
                ErrorCode::InvalidNumber,
                format!("{path}[{index}]"),
                format!("value must be finite and within ±{maximum}"),
            ));
        }
    }
    Ok(())
}
