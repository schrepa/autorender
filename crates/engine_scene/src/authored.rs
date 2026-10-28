use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use engine_core::bevy_ecs::prelude::{Entity, World};
use engine_core::collections::HashMap;
use engine_core::{ActiveScene, AuthoredIdentity, AuthoredScene};

use crate::{SceneEntity, SceneInstance};

struct Active {
    id: String,
    streamed: bool,
    entities: Arc<BTreeMap<String, Entity>>,
    ids: OnceLock<HashMap<Entity, String>>,
}

impl Active {
    fn ids(&self) -> &HashMap<Entity, String> {
        self.ids.get_or_init(|| {
            self.entities
                .iter()
                .map(|(id, &entity)| (entity, id.clone()))
                .collect()
        })
    }
}

struct Authored(Option<Active>);

pub fn active_scene(scene: Option<&SceneInstance>) -> ActiveScene {
    ActiveScene(Box::new(Authored(scene.map(|scene| Active {
        id: scene.id().to_owned(),
        streamed: scene.streamed().is_some(),
        entities: scene.shared_entities(),
        ids: OnceLock::new(),
    }))))
}

impl SceneInstance {
    pub fn authored_entity(&self, world: &World, id: &str) -> Option<Entity> {
        self.entity(id).or_else(|| {
            self.streamed()?;
            streamed_entity(world, self.id(), id)
        })
    }
}

fn streamed_entity(world: &World, scene: &str, id: &str) -> Option<Entity> {
    world
        .try_query::<(Entity, &SceneEntity)>()?
        .iter(world)
        .find(|(_, metadata)| metadata.authored_id == id && metadata.scene_id == scene)
        .map(|(entity, _)| entity)
}

impl AuthoredScene for Authored {
    fn entity_count(&self) -> Option<usize> {
        self.0.as_ref().map(|scene| scene.entities.len())
    }

    fn entity(&self, world: &World, id: &str) -> Option<Entity> {
        let scene = self.0.as_ref()?;
        scene.entities.get(id).copied().or_else(|| {
            scene
                .streamed
                .then(|| streamed_entity(world, &scene.id, id))
                .flatten()
        })
    }

    fn identity(&self, world: &World, entity: Entity) -> AuthoredIdentity {
        let scene = self.0.as_ref();
        let id = scene
            .and_then(|scene| scene.ids().get(&entity))
            .map(String::as_str);
        let metadata = world.get::<SceneEntity>(entity);
        let streamed = id.is_none()
            && metadata
                .zip(scene)
                .is_some_and(|(metadata, scene)| scene.streamed && metadata.scene_id == scene.id);
        AuthoredIdentity {
            id: id
                .or_else(|| metadata.map(|metadata| metadata.authored_id.as_str()))
                .map(str::to_owned),
            scene: if id.is_some() {
                scene.map(|scene| scene.id.as_str())
            } else {
                metadata.map(|metadata| metadata.scene_id.as_str())
            }
            .map(str::to_owned),
            name: metadata.map(|metadata| metadata.name.clone()),
            in_active_scene: id.is_some() || streamed,
        }
    }
}
