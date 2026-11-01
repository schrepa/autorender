use bevy_ecs::prelude::World;

use crate::{Camera, FrameTime, RenderSnapshot};

pub(crate) fn extract_lighting(world: &World, snapshot: &mut RenderSnapshot) {
    let eye = world.resource::<Camera>().eye();
    snapshot.camera_position = eye;
    snapshot.detail_eye = crate::TerrainDebug::detail_eye(world);
    snapshot.elapsed_seconds = world
        .get_resource::<FrameTime>()
        .map_or(0.0, |time| time.elapsed_seconds as f32);
    snapshot.presented_seconds = world
        .get_resource::<crate::PresentedTime>()
        .map_or(0.0, |time| time.seconds);
}
