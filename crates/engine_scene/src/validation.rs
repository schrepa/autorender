use engine_settings::{ErrorCode, Expected};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bevy_ecs::entity::Entity;
use engine_core::{
    Asset, AssetId, Assets, Error, Material, MeshFile, SceneContext, SceneCx, SceneEntity,
    Texture, Transform, valid_uri,
};

use crate::components::Component;
use crate::hierarchy::{ExpandedObject, ExpandedScene};
use crate::texture::TextureSource;
use crate::{SceneDocument, SceneKinds};

#[derive(Clone, Copy)]
pub(crate) enum Pick<'a> {
    Checked,
    Variant(Option<&'a str>),
}

impl<'a> Pick<'a> {
    fn variant(self) -> Option<&'a str> {
        match self {
            Self::Checked => None,
            Self::Variant(variant) => variant,
        }
    }

    fn file(self, uri: &str) -> Option<&str> {
        match self {
            Self::Checked => None,
            Self::Variant(_) => Some(uri),
        }
    }
}

pub(crate) struct Landing {
    pub entity: Entity,
    pub parent: Option<Entity>,
    pub transform: Transform,
    pub scene: SceneEntity,
    pub components: Vec<Component>,
}

impl SceneDocument {
    pub fn validate(&self, kinds: &SceneKinds) -> Result<usize, Error> {
        let expanded = self.expand()?;
        let count = expanded.entities.len();
        let entities = Arc::new(
            expanded
                .entities
                .iter()
                .map(|entity| (entity.source.id.clone(), Entity::PLACEHOLDER))
                .collect(),
        );
        let mut assets = kinds.scratch();
        self.prepare(kinds, &mut assets, expanded, &entities, Pick::Checked)?;
        Ok(count)
    }

    pub(crate) fn prepare(
        &self,
        kinds: &SceneKinds,
        assets: &mut Assets,
        expanded: ExpandedScene,
        entities: &Arc<BTreeMap<String, Entity>>,
        pick: Pick<'_>,
    ) -> Result<Vec<Vec<Landing>>, Error> {
        self.declare(assets, pick)?;
        let context = SceneContext {
            entities: Arc::clone(entities),
        };
        let cx = SceneCx::new(Some(&context));
        if let Some(environment) = &self.environment {
            assets.collect(0, "", || {
                environment.keys().try_for_each(|key| {
                    let Some(section) = kinds.section(key) else {
                        return Ok(());
                    };
                    let path = format!("$.environment.{key}");
                    let value = &environment.sections[key];
                    section.check(value, &path)?;
                    section.check_scene(value, &path, &context)
                })
            })?;
        }
        let names: Vec<String> = assets
            .groups()
            .into_iter()
            .map(|group| group.name)
            .collect();
        let joined = joined(&expanded.entities, &names);
        let mut groups: Vec<Vec<Landing>> = names.iter().map(|_| Vec::new()).collect();
        let listed = expanded.entities.into_iter().zip(expanded.transforms);
        for ((object, transform), group) in listed.zip(joined) {
            let within = if object.source.cooked {
                self.folder.as_str()
            } else {
                ""
            };
            let components = assets.collect(group, within, || {
                kinds.prepare(&object.source, &object.path, &cx)
            })?;
            let source = &object.source;
            groups[group].push(Landing {
                entity: entities[&source.id],
                parent: source.parent.as_ref().map(|parent| entities[parent]),
                transform,
                scene: object.scene_entity(&self.id),
                components,
            });
        }
        Ok(groups)
    }

    fn declare(&self, assets: &mut Assets, pick: Pick<'_>) -> Result<(), Error> {
        let mut textures: BTreeMap<&str, Asset<Texture>> = BTreeMap::new();
        for (index, texture) in self.textures.iter().enumerate() {
            let path = format!("$.textures[{index}]");
            let source = texture.source(&path, pick.variant())?;
            let file = match (&source, pick) {
                (TextureSource::File(uri), _) => pick.file(uri),
                (TextureSource::Unpicked, Pick::Variant(_)) => return Err(texture.unpicked(&path)),
                (TextureSource::Made(_) | TextureSource::Unpicked, _) => None,
            };
            let asset = assets
                .declare::<Texture>(&texture.id, file)
                .map_err(|error| error.within(&path))?;
            if let TextureSource::Made(made) = source {
                assets
                    .make(asset, made, Vec::new())
                    .map_err(|error| error.within(&path))?;
            }
            textures.insert(texture.id.as_str(), asset);
        }
        for (index, mesh) in self.meshes.iter().enumerate() {
            let path = format!("$.meshes[{index}]");
            valid_uri(&mesh.uri, &format!("{path}.uri"))?;
            assets
                .declare::<MeshFile>(&mesh.id, pick.file(&mesh.uri))
                .map_err(|error| error.within(&path))?;
        }
        let known: BTreeSet<&str> = textures.keys().copied().collect();
        for (index, material) in self.materials.iter().enumerate() {
            let path = format!("$.materials[{index}]");
            for (field, texture) in material.texture_ids() {
                reference(texture, &known, &format!("{path}.{field}"))?;
            }
            let made = material.prepare(&path)?.material(|id| textures[id]);
            let needs = made.textures().map(AssetId::from).collect();
            let asset = assets
                .declare::<Material>(&material.id, None)
                .map_err(|error| error.within(&path))?;
            assets
                .make(asset, made, needs)
                .map_err(|error| error.within(&path))?;
        }
        Ok(())
    }
}

fn joined(entities: &[ExpandedObject], names: &[String]) -> Vec<usize> {
    let last = names.len() - 1;
    let own: Vec<usize> = entities
        .iter()
        .map(|entity| {
            names
                .iter()
                .position(|name| entity.source.tags.contains(name))
                .unwrap_or(last)
        })
        .collect();
    let at: BTreeMap<&str, usize> = entities
        .iter()
        .enumerate()
        .map(|(index, entity)| (entity.source.id.as_str(), index))
        .collect();
    let parent = |index: usize| {
        let parent = entities[index].source.parent.as_deref()?;
        at.get(parent).copied()
    };
    let mut group: Vec<Option<usize>> = vec![None; entities.len()];
    let mut chain = Vec::new();
    for start in 0..entities.len() {
        let mut cursor = Some(start);
        while let Some(index) = cursor.filter(|&index| group[index].is_none()) {
            chain.push(index);
            cursor = parent(index);
        }
        let mut above = cursor.and_then(|index| group[index]).unwrap_or(0);
        while let Some(index) = chain.pop() {
            above = above.max(own[index]);
            group[index] = Some(above);
        }
    }
    group.into_iter().flatten().collect()
}

pub fn identifier(id: &str, path: &str) -> Result<(), Error> {
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

pub fn unknown<S: Into<String>>(
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
