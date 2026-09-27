use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use engine_core::{
    AnimationClip, LocalLight, Mesh, ParticleEmitter, Texture, Tint, Transform, glam::Mat4,
};

use crate::{
    MeshSource, SCENE_VERSION, SceneDocument, SceneError, SceneFile, SceneFileKind, SceneFiles,
    SceneObject, SceneTransform,
    animation::{PreparedPlayer, PreparedSkin},
    hierarchy::ExpandedObject,
    material::PreparedMaterial,
};

pub(crate) struct PreparedScene {
    pub environment: Option<crate::EnvironmentSettings>,
    pub colliders: Vec<Option<engine_collision::Collider>>,
    pub lights: Vec<Option<LocalLight>>,
    pub emitters: Vec<Option<ParticleEmitter>>,
    pub tints: Vec<Option<Tint>>,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<PreparedMaterial>,
    pub textures: Vec<Texture>,
    pub transforms: Vec<Transform>,
    pub entities: Vec<ExpandedObject>,
    pub clips: Vec<AnimationClip>,
    pub players: Vec<Option<PreparedPlayer>>,
    pub skin_binds: Vec<Arc<[Mat4]>>,
    pub skins: Vec<Option<PreparedSkin>>,
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
            if let MeshSource::Buffer { uri } = &mesh.source {
                add(uri, SceneFileKind::Buffer);
            }
        }
        for animation in &self.animations {
            if let Some(uri) = &animation.uri {
                add(uri, SceneFileKind::Buffer);
            }
        }
        files
    }

    /// `files`: `None` validates structure only, reading referenced files as
    /// placeholders.
    pub(crate) fn prepare(&self, files: Option<&SceneFiles>) -> Result<PreparedScene, SceneError> {
        if !(1..=SCENE_VERSION).contains(&self.version) {
            return Err(SceneError::new(
                "unsupported_version",
                "$.version",
                format!(
                    "expected version 1 through {SCENE_VERSION}; found {}",
                    self.version
                ),
            ));
        }
        if self.version < 8
            && let Some((index, field)) =
                self.materials
                    .iter()
                    .enumerate()
                    .find_map(|(index, material)| {
                        material.version_eight_field().map(|field| (index, field))
                    })
        {
            return Err(SceneError::new(
                "version_required",
                format!("$.materials[{index}].{field}"),
                "glTF material maps, texture settings, emissive colors, alpha modes and double-sided materials require scene version 8; set version to 8 and increment revision",
            ));
        }
        if self.version < 8 {
            let tinted = |entity: &crate::SceneObject| entity.tint.is_some();
            let path = if let Some(index) = self.entities.iter().position(tinted) {
                Some(format!("$.entities[{index}].tint"))
            } else {
                self.prefabs
                    .iter()
                    .enumerate()
                    .find_map(|(prefab, definition)| {
                        definition
                            .entities
                            .iter()
                            .position(tinted)
                            .map(|index| format!("$.prefabs[{prefab}].entities[{index}].tint"))
                    })
            };
            if let Some(path) = path {
                return Err(SceneError::new(
                    "version_required",
                    path,
                    "object tints require scene version 8; set version to 8 and increment revision",
                ));
            }
            if let Some(index) = self
                .meshes
                .iter()
                .position(|mesh| mesh.source.has_vertex_extras())
            {
                return Err(SceneError::new(
                    "version_required",
                    format!("$.meshes[{index}].source.vertices"),
                    "vertex colors, second texture coordinates, joints and weights require scene version 8; set version to 8 and increment revision",
                ));
            }
            let animated =
                |entity: &SceneObject| entity.animation.is_some() || entity.skin.is_some();
            let path = if !self.animations.is_empty() {
                Some("$.animations".to_owned())
            } else if !self.skins.is_empty() {
                Some("$.skins".to_owned())
            } else if let Some(index) = self.entities.iter().position(animated) {
                Some(format!("$.entities[{index}]"))
            } else {
                self.prefabs
                    .iter()
                    .enumerate()
                    .find_map(|(prefab, definition)| {
                        definition
                            .entities
                            .iter()
                            .position(animated)
                            .map(|index| format!("$.prefabs[{prefab}].entities[{index}]"))
                    })
            };
            if let Some(path) = path {
                return Err(SceneError::new(
                    "version_required",
                    path,
                    "animations and skins require scene version 8; set version to 8 and increment revision",
                ));
            }
        }
        if self.version < 7 {
            let version_seven = |path: String| {
                SceneError::new(
                    "version_required",
                    path,
                    "image and buffer files require scene version 7; set version to 7 and increment revision",
                )
            };
            if let Some(index) = self
                .textures
                .iter()
                .position(|texture| texture.uri.is_some())
            {
                return Err(version_seven(format!("$.textures[{index}].uri")));
            }
            if let Some(index) = self
                .meshes
                .iter()
                .position(|mesh| matches!(mesh.source, MeshSource::Buffer { .. }))
            {
                return Err(version_seven(format!("$.meshes[{index}].source")));
            }
        }
        if self.version < 6 {
            let version_six = |path: String, what: &str| {
                SceneError::new(
                    "version_required",
                    path,
                    format!(
                        "{what} require scene version 6; set version to 6 and increment revision"
                    ),
                )
            };
            if let Some(section) = self
                .environment
                .as_ref()
                .and_then(|environment| environment.version_six_field())
            {
                return Err(version_six(
                    format!("$.environment.{section}"),
                    "grading, shadow cascade, local light, particle, sun shaft and reflection settings",
                ));
            }
            if let Some(index) = self
                .materials
                .iter()
                .position(|material| material.opacity.is_some())
            {
                return Err(version_six(
                    format!("$.materials[{index}].opacity"),
                    "material opacities",
                ));
            }
            if let Some(index) = self
                .entities
                .iter()
                .position(|entity| entity.light.is_some() || entity.particles.is_some())
            {
                return Err(version_six(
                    format!("$.entities[{index}]"),
                    "entity lights and particles",
                ));
            }
            if self
                .prefabs
                .iter()
                .flat_map(|prefab| &prefab.entities)
                .any(|entity| entity.light.is_some() || entity.particles.is_some())
            {
                return Err(version_six(
                    "$.prefabs".to_owned(),
                    "entity lights and particles",
                ));
            }
        }
        if self.version < 5 && self.environment.is_some() {
            return Err(SceneError::new(
                "version_required",
                "$.environment",
                "scene environments require scene version 5; set version to 5 and increment revision",
            ));
        }
        if self.version < 5
            && let Some(index) = self
                .materials
                .iter()
                .position(|material| material.surface.is_some())
        {
            return Err(SceneError::new(
                "version_required",
                format!("$.materials[{index}].surface"),
                "material surfaces require scene version 5; set version to 5 and increment revision",
            ));
        }
        if self.version < 4
            && self
                .entities
                .iter()
                .chain(self.prefabs.iter().flat_map(|prefab| &prefab.entities))
                .any(|entity| entity.collider.is_some())
        {
            return Err(SceneError::new(
                "version_required",
                "$.version",
                "colliders require scene version 4; set version to 4 and increment revision",
            ));
        }
        if self.version == 1 && (!self.textures.is_empty() || self.materials.iter().any(|material| material.texture.is_some())
            || self.meshes.iter().any(|mesh| matches!(&mesh.source, MeshSource::Inline {vertices,..} if vertices.iter().any(|vertex| vertex.uv != [0.0;2])))) {
            return Err(SceneError::new("version_required", "$.version", "textures and UV data require scene version 2; set version to 2 and increment revision"));
        }
        if self.version < 3
            && (!self.prefabs.is_empty()
                || self.entities.iter().any(|entity| {
                    entity.parent.is_some()
                        || entity.prefab.is_some()
                        || entity.mesh.is_none()
                        || entity.material.is_none()
                }))
        {
            return Err(SceneError::new(
                "version_required",
                "$.version",
                "hierarchies, prefabs and transform-only nodes require scene version 3; set version to 3 and increment revision",
            ));
        }
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
        let players = expanded
            .entities
            .iter()
            .map(|entity| {
                let Some(player) = &entity.source.animation else {
                    return Ok(None);
                };
                let path = format!("{}.animation", entity.path);
                reference(&player.clip, &animation_ids, &format!("{path}.clip"))?;
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
            })
            .collect::<Result<Vec<_>, _>>()?;
        let skins = expanded
            .entities
            .iter()
            .map(|entity| {
                let Some(id) = &entity.source.skin else {
                    return Ok(None);
                };
                let path = format!("{}.skin", entity.path);
                reference(id, &skin_ids, &path)?;
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
            })
            .collect::<Result<Vec<_>, _>>()?;
        let colliders = expanded
            .entities
            .iter()
            .map(|entity| {
                entity
                    .source
                    .collider
                    .map(|collider| collider.runtime(&format!("{}.collider", entity.path)))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let lights = expanded
            .entities
            .iter()
            .map(|entity| {
                entity
                    .source
                    .light
                    .map(|light| light.runtime(&format!("{}.light", entity.path)))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let emitters = expanded
            .entities
            .iter()
            .map(|entity| {
                entity
                    .source
                    .particles
                    .map(|emitter| emitter.runtime(&format!("{}.particles", entity.path)))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;

        let tints = expanded
            .entities
            .iter()
            .map(|entity| {
                entity
                    .source
                    .tint
                    .map(|tint| {
                        if tint.iter().all(|factor| (0.0..=1.0).contains(factor)) {
                            Ok(Tint(tint))
                        } else {
                            Err(SceneError::new(
                                "invalid_tint",
                                format!("{}.tint", entity.path),
                                "tint factors must be finite and in [0, 1]",
                            ))
                        }
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Build every CPU asset before acquiring a mutable world resource.
        // Invalid late assets cannot leave earlier ones resident.
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
        for (entity, skin) in expanded.entities.iter().zip(&skins) {
            let (Some(skin), Some(mesh)) = (skin, &entity.source.mesh) else {
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
        let clips = self
            .animations
            .iter()
            .enumerate()
            .map(|(index, animation)| animation.clip(&format!("$.animations[{index}]"), files))
            .collect::<Result<Vec<_>, _>>()?;
        let environment = self
            .environment
            .map(|environment| environment.resolve("$.environment"))
            .transpose()?;
        Ok(PreparedScene {
            environment,
            colliders,
            lights,
            emitters,
            tints,
            meshes,
            materials,
            textures,
            transforms: expanded.transforms,
            entities: expanded.entities,
            clips,
            players,
            skin_binds,
            skins,
        })
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
