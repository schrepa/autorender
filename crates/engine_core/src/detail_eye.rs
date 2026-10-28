use bevy_ecs::prelude::{Resource, World};
use glam::Vec3;

use crate::Camera;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct TerrainDebug {
    pub freeze: Option<Vec3>,
}

impl TerrainDebug {
    pub(crate) fn detail_eye(world: &World) -> Vec3 {
        world
            .get_resource::<Self>()
            .and_then(|debug| debug.freeze)
            .unwrap_or_else(|| world.resource::<Camera>().eye())
    }
}
