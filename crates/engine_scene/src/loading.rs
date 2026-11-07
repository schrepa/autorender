use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use engine_assets::SceneAssets;
use engine_core::{SceneBuffers, SceneContext};

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

pub(crate) fn context(
    document: &SceneDocument,
    assets: Arc<SceneAssets>,
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
        components: Arc::new(
            document
                .entities
                .iter()
                .map(|entity| {
                    let authored = entity
                        .authored()
                        .map(|(key, value)| (key.to_owned(), value.clone()))
                        .collect();
                    (entity.id.clone(), authored)
                })
                .collect(),
        ),
        entities,
        kinds: kinds.listed_kinds().into(),
        list: kinds.job_list(),
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
