use std::sync::Arc;

use bevy_ecs::prelude::{Entity, Resource};
use engine_place::Slot;
use glam::Vec3;

pub trait SceneQuery: Send + Sync + 'static {
    fn sweep(&self, radius: f32, from: Vec3, along: Vec3, ignore: Entity) -> Option<f32>;
}

struct Empty;

impl SceneQuery for Empty {
    fn sweep(&self, _: f32, _: Vec3, _: Vec3, _: Entity) -> Option<f32> {
        None
    }
}

#[derive(Resource)]
pub struct SceneQueries(pub Arc<dyn SceneQuery>);

impl Default for SceneQueries {
    fn default() -> Self {
        Self(Arc::new(Empty))
    }
}

impl Slot for SceneQueries {
    const NAME: &'static str = "scene_queries";
}
