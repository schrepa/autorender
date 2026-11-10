use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, Background, Drawn, Error, Files, Landed, Refusals, SceneCx, Spawner, Transform,
    valid_uri,
};
use engine_places_rendering::animation::{Animator, Model, Start};
use serde_json::Value;

use crate::AnimationBrick;
use crate::import::{ImportShared, import_glb};

pub const MODEL_PART: &str = "model";

#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct ModelLoaded {
    pub parts: Vec<Entity>,
    pub animator: Option<Entity>,
}

type Read = Background<Arc<[u8]>, String>;

enum Stage {
    Held(Arc<[u8]>),
    Reading(Read),
    Loaded(ModelLoaded),
    Refused { _failure: Read },
}

#[derive(Resource, Default)]
pub(crate) struct ModelLoads {
    models: BTreeMap<Entity, Stage>,
    shared: ImportShared,
}

pub(crate) fn check_model(model: &Model, path: &str) -> Result<(), Error> {
    valid_uri(&model.file, &format!("{path}.file"))
}

pub(crate) fn model_files(model: &Model) -> Vec<String> {
    vec![model.file.clone()]
}

pub(crate) fn prepare_model(
    mut model: Model,
    cx: &SceneCx<'_>,
    path: &str,
) -> Result<Model, Error> {
    model.bytes = cx
        .buffer(&model.file, &format!("{path}.file"))
        .ok()
        .flatten();
    Ok(model)
}

pub(crate) fn model_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(Model)).unwrap_or(Value::Null)
}

fn bytes_of(bytes: &[u8]) -> Result<Arc<[u8]>, Error> {
    Ok(Arc::from(bytes))
}

fn drop_parts(
    loaded: &ModelLoaded,
    nodes: &Query<(), With<Transform>>,
    spawner: &mut Spawner<AnimationBrick>,
) {
    for &part in loaded.parts.iter().chain(&loaded.animator) {
        if nodes.contains(part) {
            spawner.despawn(part);
        }
    }
}

fn spawn_parts(
    entity: Entity,
    imported: crate::import::Model,
    play: Option<&Start>,
    spawner: &mut Spawner<AnimationBrick>,
) -> ModelLoaded {
    let parts: Vec<Entity> = imported
        .statics
        .iter()
        .map(|&(mesh, material)| {
            spawner.spawn((
                Transform::default(),
                Drawn::new(engine_places::meshes::MESH, mesh, material),
            ))
        })
        .collect();
    let animator = imported.rig.map(|rig| {
        let animator = Animator::new(rig);
        let animator = match play {
            Some(start) => animator.starting(start.clone()),
            None => animator,
        };
        spawner.spawn((Transform::default(), animator))
    });
    spawner.set_parents(
        parts
            .iter()
            .chain(&animator)
            .map(|&part| (part, Some(entity)))
            .collect(),
    );
    ModelLoaded { parts, animator }
}

pub(crate) fn load_models(
    models: Query<(Entity, Ref<Model>)>,
    nodes: Query<(), With<Transform>>,
    (mut loads, mut files, mut assets): (ResMut<ModelLoads>, ResMut<Files>, ResMut<AssetLibrary>),
    mut refusals: ResMut<Refusals>,
    mut spawner: Spawner<AnimationBrick>,
) {
    let ModelLoads {
        models: held,
        shared,
    } = &mut *loads;
    for (entity, model) in &models {
        if !model.is_changed() && held.contains_key(&entity) {
            continue;
        }
        if let Some(Stage::Loaded(loaded)) = held.remove(&entity) {
            drop_parts(&loaded, &nodes, &mut spawner);
        }
        let stage = match &model.bytes {
            Some(bytes) => Stage::Held(Arc::clone(bytes)),
            None => Stage::Reading(files.read(&model.file, bytes_of)),
        };
        held.insert(entity, stage);
    }
    for (&entity, stage) in held.iter_mut() {
        let bytes = match stage {
            Stage::Held(bytes) => Ok(Arc::clone(bytes)),
            Stage::Reading(read) => match read.poll(&*files) {
                Landed::Waiting => continue,
                Landed::Ready(bytes) => Ok(Arc::clone(bytes)),
                Landed::Failed(error) => Err(error.clone()),
            },
            Stage::Loaded(_) | Stage::Refused { .. } => continue,
        };
        let Ok((_, model)) = models.get(entity) else {
            continue;
        };
        let imported = match bytes {
            Ok(bytes) => import_glb(&bytes, &mut assets, shared),
            Err(error) => Err(error),
        };
        *stage = match imported {
            Ok(imported) => {
                let loaded = spawn_parts(entity, imported, model.play.as_ref(), &mut spawner);
                spawner.insert(entity, loaded.clone());
                refusals.accept(entity, MODEL_PART);
                Stage::Loaded(loaded)
            }
            Err(error) => {
                let error = error.within("$.model");
                refusals.refuse(entity, MODEL_PART, error.clone());
                Stage::Refused {
                    _failure: Background::failed(error),
                }
            }
        };
    }
}

pub(crate) fn release(
    In(entity): In<Entity>,
    mut loads: ResMut<ModelLoads>,
    nodes: Query<(), With<Transform>>,
    mut refusals: ResMut<Refusals>,
    mut spawner: Spawner<AnimationBrick>,
) {
    refusals.accept(entity, MODEL_PART);
    if let Some(Stage::Loaded(loaded)) = loads.models.remove(&entity) {
        drop_parts(&loaded, &nodes, &mut spawner);
    }
}
