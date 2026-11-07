use bevy_ecs::prelude::Component;
use engine_place::{ErrorCode, List, MeshHandle};
use glam::{Mat4, Vec3};
use serde_json::Value;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamingFocus;

pub struct StreamedKeys;

impl List for StreamedKeys {
    const NAME: &'static str = "streamed_keys";
    type Member = StreamedKey;
}

#[derive(Clone, Copy)]
pub struct StreamedKey {
    pub key: &'static str,
    pub streams: Streams,
}

#[derive(Clone, Copy)]
pub enum Streams {
    Reaching(fn(&Value, &StreamedAt<'_>) -> f32),
    Never(ErrorCode),
}

pub struct StreamedAt<'a> {
    pub world: Mat4,
    pub position: Vec3,
    pub from: f32,
    pub mesh_sizes: &'a dyn Fn(MeshHandle) -> f32,
}

impl StreamedAt<'_> {
    pub fn mesh_size(&self, mesh: MeshHandle) -> f32 {
        (self.mesh_sizes)(mesh)
    }
}
