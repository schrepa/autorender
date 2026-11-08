use std::ops::Range;

use bevy_ecs::prelude::*;
use engine_core::{
    Detail, DeviceLimits, DrawnTable, MaterialHandle, MeshHandle, PlanState, Views,
};
use engine_render::slots::DRAWN_SLOTS;

use crate::presentation::{frustum_uniform, rasterized};
use crate::settings::{Culling, OcclusionCulling, SubpixelCulling};
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

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct MeshPlan {
    pub culling: bool,
    pub occlusion_enabled: bool,
    pub occlusion: bool,
    pub subpixel_enabled: bool,
    pub subpixel: bool,
    pub budget: u32,
    pub lod_levels: bool,
}

pub(crate) fn plan_draws(
    table: Res<DrawnTable>,
    (views, detail): (Res<Views>, Res<Detail>),
    (culling, occlusion, subpixel): (Res<Culling>, Res<OcclusionCulling>, Res<SubpixelCulling>),
    mut decided: ResMut<MeshPlan>,
) {
    let bounded = views.main().is_some_and(|main| {
        let (planes, absolute) = frustum_uniform(rasterized(main));
        absolute.iter().flatten().all(|value| {
            value.is_finite() && *value <= engine_core::presentation_bounds::MAX_COEFFICIENT
        }) && planes.iter().flatten().all(|value| value.is_finite())
    });
    let frustum = culling.frustum && bounded;
    let candidates = (table.census(MESH).drawn + table.census(SKIN).drawn) as usize;
    let unbudgeted = frustum && culling.budget() >= candidates;
    decided.set_if_neq(MeshPlan {
        culling: culling.frustum,
        occlusion_enabled: occlusion.enabled,
        occlusion: occlusion.enabled && unbudgeted && candidates != 0,
        subpixel_enabled: subpixel.enabled,
        subpixel: subpixel.enabled && unbudgeted && candidates != 0,
        budget: culling.budget().min(u32::MAX as usize) as u32,
        lod_levels: detail.enabled,
    });
}
