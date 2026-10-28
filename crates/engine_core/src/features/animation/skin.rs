use std::sync::Arc;

use bevy_ecs::{
    change_detection::Ref,
    prelude::{Component, Entity, World},
    query::QueryState,
};
use glam::Mat4;

use crate::ticks::ScanClock;
use crate::{MeshHandle, TransformIndex};

#[derive(Component, Debug, Clone, PartialEq)]
pub struct Skin {
    pub joints: Vec<Entity>,
    pub inverse_binds: Arc<[Mat4]>,
}

impl Skin {
    pub fn new(joints: Vec<Entity>, inverse_binds: impl Into<Arc<[Mat4]>>) -> Self {
        Self {
            joints,
            inverse_binds: inverse_binds.into(),
        }
    }

    pub fn inverse_bind(&self, index: usize) -> Mat4 {
        self.inverse_binds
            .get(index)
            .copied()
            .unwrap_or(Mat4::IDENTITY)
    }
}

pub(crate) struct SkinExtraction {
    pub(crate) query: QueryState<(Entity, Ref<'static, Skin>)>,
    pub(crate) hints: Vec<TransformIndex>,
    pub(crate) joints: usize,
    pub(crate) skinned: Vec<usize>,
    pub(crate) scratch: Vec<(usize, u32)>,
    pub(crate) inputs: Vec<(Entity, usize, Option<MeshHandle>)>,
    pub(crate) key: Option<((u64, u64), u64, u64)>,
    pub(crate) scans: ScanClock,
}

impl SkinExtraction {
    pub(crate) fn new(world: &mut World) -> Self {
        Self {
            query: world.query::<(Entity, Ref<Skin>)>(),
            hints: Vec::new(),
            joints: 0,
            skinned: Vec::new(),
            scratch: Vec::new(),
            inputs: Vec::new(),
            key: None,
            scans: ScanClock::default(),
        }
    }
}
