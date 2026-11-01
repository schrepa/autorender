use bevy_ecs::prelude::Component;
use engine_place::{MaterialHandle, MeshHandle};
use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplineMesh {
    pub points: Vec<Vec3>,
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
}
