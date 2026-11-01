use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_assets::{Aabb, SceneAssets};
use engine_core::{Error, Mesh, SceneBuffers, SceneContext, ScenePath};

use crate::{SceneDocument, SceneKinds};

pub(crate) fn named(document: &SceneDocument) -> Arc<SceneAssets> {
    Arc::new(SceneAssets {
        meshes: document
            .meshes
            .iter()
            .map(|mesh| (mesh.id.clone(), Default::default()))
            .collect(),
        materials: document
            .materials
            .iter()
            .map(|material| (material.id.clone(), Default::default()))
            .collect(),
        textures: document
            .textures
            .iter()
            .map(|texture| (texture.id.clone(), Default::default()))
            .collect(),
    })
}

pub(crate) fn paths(document: &SceneDocument, kinds: &SceneKinds) -> Result<Vec<ScenePath>, Error> {
    let cx = kinds.cx(None, None);
    named(document).lenient(|| {
        let mut paths = Vec::new();
        for (index, entity) in document.entities.iter().enumerate() {
            let answered = kinds.paths(entity, &format!("$.entities[{index}]"), &cx)?;
            paths.extend(answered.into_iter().map(|path| ScenePath {
                terrain: entity.id.clone(),
                ..path
            }));
        }
        Ok(paths)
    })
}

pub(crate) fn bounds(document: &SceneDocument, meshes: &[Mesh]) -> BTreeMap<String, Aabb> {
    document
        .meshes
        .iter()
        .zip(meshes)
        .map(|(source, mesh)| (source.id.clone(), mesh.bounds()))
        .collect()
}

pub(crate) fn context(
    document: &SceneDocument,
    (assets, bounds): (Arc<SceneAssets>, BTreeMap<String, Aabb>),
    paths: Vec<ScenePath>,
    entities: Arc<BTreeMap<String, Entity>>,
    kinds: &SceneKinds,
    files: Option<Arc<SceneBuffers>>,
) -> SceneContext {
    SceneContext {
        id: document.id.clone(),
        assets,
        prefabs: Arc::new(
            document
                .prefabs
                .iter()
                .enumerate()
                .map(|(index, prefab)| {
                    let value = serde_json::to_value(prefab).expect("prefab_serializes");
                    (prefab.id.clone(), (index, value))
                })
                .collect(),
        ),
        paths: paths.into(),
        entities,
        kinds: kinds.listed_kinds().into(),
        bounds: Arc::new(bounds),
        files,
    }
}

pub(crate) fn kept(world: &World) -> Option<SceneContext> {
    world.get_resource::<SceneContext>().cloned()
}

pub(crate) fn within<R>(context: Option<&SceneContext>, run: impl FnOnce() -> R) -> R {
    match context {
        Some(context) => context.scope(run),
        None => run(),
    }
}
