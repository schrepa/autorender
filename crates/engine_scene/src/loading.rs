use std::sync::Arc;

use engine_assets::SceneAssets;
use engine_core::SceneContext;

use crate::SceneDocument;

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

pub(crate) fn within<R>(context: Option<&SceneContext>, run: impl FnOnce() -> R) -> R {
    match context {
        Some(context) => context.scope(run),
        None => run(),
    }
}
