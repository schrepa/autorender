use std::ops::Range;

use engine_core::{DeviceLimits, DrawnTable, MaterialHandle, MeshHandle, PlanState};
use engine_render::slots::DRAWN_SLOTS;

use crate::uploads::PoseData;
use crate::{MESH, SKIN};

const INITIAL_CAPACITY: usize = 1_024;

pub(crate) fn capacity_for(needed: usize, limit: usize) -> usize {
    needed.max(INITIAL_CAPACITY).next_power_of_two().min(limit)
}

pub(crate) fn instance_limit(limits: &DeviceLimits) -> usize {
    let record = size_of::<PoseData>().max(6 * size_of::<u32>()) as u64;
    if limits.storage_room == 0 {
        return DRAWN_SLOTS;
    }
    (limits.storage_room / record).min(DRAWN_SLOTS as u64) as usize
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BatchLook {
    pub masked: bool,
    pub double_sided: bool,
    pub extended: bool,
    pub skinned: bool,
}

#[derive(Debug, PartialEq)]
pub(crate) struct DrawBatch {
    pub mesh: MeshHandle,
    pub material: Option<MaterialHandle>,
    pub instances: Range<u32>,
    pub skinned: bool,
}

#[derive(Default)]
pub(crate) struct MeshBatches {
    pub ordered: Vec<u32>,
    pub batches: Vec<DrawBatch>,
    generation: Option<u64>,
}

impl MeshBatches {
    pub(crate) fn generation(&self) -> u64 {
        self.generation.unwrap_or(0)
    }

    pub(crate) fn settle(&mut self, table: &DrawnTable) -> bool {
        let PlanState::Built { generation } = table.plan() else {
            return false;
        };
        if self.generation == Some(generation) {
            return false;
        }
        self.generation = Some(generation);
        self.ordered.clear();
        self.batches.clear();
        for batch in table
            .batches()
            .iter()
            .filter(|batch| batch.kind == MESH || batch.kind == SKIN)
        {
            let Some(mesh) = batch.geometry else {
                continue;
            };
            let start = self.ordered.len() as u32;
            let range = batch.instances.start as usize..batch.instances.end as usize;
            self.ordered.extend_from_slice(&table.ordered()[range]);
            self.batches.push(DrawBatch {
                mesh,
                material: batch.material,
                instances: start..self.ordered.len() as u32,
                skinned: batch.kind == SKIN,
            });
        }
        true
    }
}
