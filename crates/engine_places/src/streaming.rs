use bevy_ecs::prelude::{Component, Resource};
use engine_place::Slot;
use glam::Vec3;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamingFocus;

#[derive(Clone, Debug, PartialEq)]
pub struct StreamingPath {
    pub layer: String,
    pub origin: Vec3,
    pub points: Vec<[f32; 2]>,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct StreamingPaths {
    pub paths: Vec<StreamingPath>,
}

impl Slot for StreamingPaths {
    const NAME: &'static str = "streaming_paths";
}
