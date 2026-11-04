use std::ops::Range;

use bevy_ecs::prelude::Resource;
use engine_codec::Expected;
use engine_core::{Error, ErrorCode, MaterialHandle, MeshHandle};
use glam::Mat4;

use crate::retained::RetainedRender;

const INITIAL_CAPACITY: usize = 1_024;

pub(crate) fn capacity_for(needed: usize, limit: usize) -> usize {
    needed.max(INITIAL_CAPACITY).next_power_of_two().min(limit)
}

#[derive(Debug, PartialEq)]
pub struct DrawBatch {
    pub mesh: MeshHandle,
    pub material: Option<MaterialHandle>,
    pub instances: Range<u32>,
}

#[derive(Resource, Default)]
pub struct BatchPlan {
    pub ordered: Vec<(u64, u32)>,
    pub batches: Vec<DrawBatch>,
    generation: u64,
    changed: bool,
    pub instance_limit: Option<usize>,
    pub instance_capacity: usize,
    pub capacity_grown: u64,
    pub refused: Option<Error>,
    valid: bool,
}

impl BatchPlan {
    pub(crate) fn limit(&self) -> usize {
        self.instance_limit
            .unwrap_or(engine_places::meshes::MOVED_SLOT as usize)
    }

    pub(crate) fn grow_capacity(&mut self, payload: usize) {
        if self.instance_capacity == 0 || payload > self.instance_capacity {
            self.instance_capacity = capacity_for(payload, self.limit());
            self.capacity_grown += 1;
        }
    }

    pub(crate) fn prepare_frame(
        &mut self,
        retained: &RetainedRender,
        view_projection: Mat4,
    ) -> Result<(), Error> {
        self.changed = false;
        if !view_projection.is_finite() {
            self.invalidate();
            return Err(
                Error::new(ErrorCode::RenderFailed, "$.snapshot.view_projection")
                    .expected(Expected::Finite),
            );
        }
        if self.valid && !retained.changed() {
            return Ok(());
        }
        let slots = retained.instances().len();
        if let Some(limit) = self.instance_limit
            && slots > limit
        {
            self.invalidate();
            return Err(Error::new(ErrorCode::RenderFailed, "$.snapshot.instances")
                .at_most(limit as u64)
                .got(slots));
        }
        self.rebuild(retained);
        Ok(())
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn changed(&self) -> bool {
        self.changed
    }

    fn invalidate(&mut self) {
        if self.valid {
            self.generation += 1;
            self.changed = true;
        }
        self.valid = false;
        self.ordered.clear();
        self.batches.clear();
    }

    fn rebuild(&mut self, retained: &RetainedRender) {
        let _profile = engine_core::profile::span("render", "batch_plan_rebuild");
        self.ordered.clear();
        self.ordered
            .extend(retained.drawn().map(|(slot, instance)| {
                let material = instance
                    .material
                    .map_or(0, |material| material.slot() as u64 + 1);
                (
                    ((instance.mesh.slot() as u64) << 32) | material,
                    slot as u32,
                )
            }));
        self.ordered.sort_unstable();
        self.batches.clear();
        let instances = retained.instances();
        let mut start = 0;
        for at in 1..=self.ordered.len() {
            if at == self.ordered.len() || self.ordered[at].0 != self.ordered[start].0 {
                let instance = &instances[self.ordered[start].1 as usize];
                self.batches.push(DrawBatch {
                    mesh: instance.mesh,
                    material: instance.material,
                    instances: start as u32..at as u32,
                });
                start = at;
            }
        }
        self.generation += 1;
        self.changed = true;
        self.valid = true;
    }
}
