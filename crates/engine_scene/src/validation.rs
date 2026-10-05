use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use engine_core::{
    AnimationClip, LocalLight, Mesh, ParticleEmitter, ReflectionProbe, Texture, Tint, Transform,
    glam::Mat4,
};

use crate::{
    MeshSource, SceneDocument, SceneError, SceneFile, SceneFileKind, SceneFiles, SceneTransform,
    animation::{PreparedPlayer, PreparedSkin},
    decal::PreparedDecal,
    hierarchy::ExpandedObject,
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

    /// The files this document references (decision 030), each once, in
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
                _ => {}
            }
        }
        for animation in &self.animations {
            if let Some(uri) = &animation.uri {
                add(uri, SceneFileKind::Buffer);
            }
        }
        // Terrains (decision 051): heightmaps as buffers, and terrain files.
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
        self.check_version()?;
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
        let expanded = self.expand(&mesh_ids, &material_ids)?;
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
        // Terrains read their files (decision 051).
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

        let CpuAssets {
            textures,
            meshes,
            materials,
        } = self.prepare_assets(files)?;
        self.check_skin_joints(&expanded.entities, &components, &meshes)?;
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
        })
    }

    /// Animation and skin sources (decisions 035 and 036): their unique IDs,
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
                    "invalid_name",
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
            .map(|(index, texture)| {
                // Texel alpha is coverage from version 8 (decision 031).
                texture.texture(&format!("$.textures[{index}]"), files, self.version >= 8)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let meshes = self
            .meshes
            .iter()
            .enumerate()
            .map(|(index, mesh)| {
                mesh.source
                    .mesh(&format!("$.meshes[{index}].source"), files, self.version)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let materials = self
            .materials
            .iter()
            .enumerate()
            .map(|(index, material)| {
                material.prepare(&format!("$.materials[{index}]"), self.version)
            })
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
                    "invalid_skin",
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
        let tint = |tint: [f32; 4]| {
            if tint.iter().all(|factor| (0.0..=1.0).contains(factor)) {
                Ok(Tint(tint))
            } else {
                Err(SceneError::new(
                    "invalid_tint",
                    format!("{path}.tint"),
                    "tint factors must be finite and in [0, 1]",
                ))
            }
        };
        Ok(PreparedEntity {
            player: self.prepare_player(entity, resolve, &rigs.animation_ids)?,
            skin: self.prepare_skin(entity, resolve, &rigs.skin_ids)?,
            collider: source
                .collider
                .map(|collider| collider.runtime(&format!("{path}.collider")))
                .transpose()?,
            light: source
                .light
                .map(|light| light.runtime(&format!("{path}.light")))
                .transpose()?,
            emitter: source
                .particles
                .map(|emitter| emitter.runtime(&format!("{path}.particles")))
                .transpose()?,
            tint: source.tint.map(tint).transpose()?,
            decal: source
                .decal
                .as_ref()
                .map(|decal| decal.prepare(&format!("{path}.decal"), &self.materials))
                .transpose()?,
            probe: source
                .reflection_probe
                .map(|probe| probe.runtime(&format!("{path}.reflection_probe")))
                .transpose()?,
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
                "invalid_animation",
                format!("{path}.speed"),
                "speed must be finite and within ±1000",
            ));
        }
        if !player.time.is_finite() || player.time < 0.0 {
            return Err(SceneError::new(
                "invalid_animation",
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
                        "unknown_reference",
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
                "invalid_skin",
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
                        "unknown_reference",
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
