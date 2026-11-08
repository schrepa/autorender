use std::sync::Arc;

use bevy_ecs::prelude::{Entity, Resource};
use engine_place::Slot;
use glam::{Quat, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CastShape {
    Sphere { radius: f32 },
    Box { half_extents: Vec3, rotation: Quat },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CastFilter {
    pub layers: u32,
    pub ignore: Option<Entity>,
}

impl Default for CastFilter {
    fn default() -> Self {
        Self {
            layers: u32::MAX,
            ignore: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CastHit {
    pub entity: Entity,
    pub distance: f32,
    pub point: Vec3,
    pub normal: Vec3,
}

pub trait SceneQuery: Send + Sync + 'static {
    fn cast(&self, shape: CastShape, from: Vec3, along: Vec3, filter: CastFilter)
    -> Option<CastHit>;
}

struct Empty;

impl SceneQuery for Empty {
    fn cast(&self, _: CastShape, _: Vec3, _: Vec3, _: CastFilter) -> Option<CastHit> {
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
