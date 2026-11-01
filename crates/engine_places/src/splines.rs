use bevy_ecs::prelude::Component;
use engine_place::{MaterialHandle, Mesh3d, MeshHandle};
use glam::Vec3;

#[derive(Component, Clone, Debug, PartialEq)]
#[require(Mesh3d)]
pub struct SplineMesh {
    pub points: Vec<Vec3>,
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
}
