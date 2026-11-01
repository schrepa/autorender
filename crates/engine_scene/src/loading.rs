use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_assets::SceneAssets;
use engine_core::{Error, SceneContext, ScenePath};

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
    named(document).scope(|| {
        let mut paths = Vec::new();
        for (index, entity) in document.entities.iter().enumerate() {
            paths.extend(kinds.paths(entity, &format!("$.entities[{index}]"))?);
        }
        Ok(paths)
    })
}

pub(crate) fn context(
    document: &SceneDocument,
    assets: Arc<SceneAssets>,
    paths: Vec<ScenePath>,
    entities: Arc<BTreeMap<String, Entity>>,
) -> SceneContext {
    SceneContext {
        id: document.id.clone(),
        assets,
        prefabs: Arc::new(
            document
                .prefabs
                .iter()
                .map(|prefab| {
                    (
                        prefab.id.clone(),
                        serde_json::to_value(prefab).expect("prefab_serializes"),
                    )
                })
                .collect(),
        ),
        paths: paths.into(),
        entities,
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
