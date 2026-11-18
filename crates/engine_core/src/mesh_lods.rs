use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_assets::{AssetLibrary, GeometryHandle, MeshLods, lod_input};
use engine_registry::{JobBatch, JobOrder, JobQueue};

use crate::jobs::{MESH_LODS_NAME, record_key};
use crate::{
    Detail, Interest, JobRef, MESH_LODS, RecordKey, Records, Request, VIEW_JOBS, content_hash,
    job_priority,
};

#[derive(Resource, Default, serde::Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MeshLevels {
    #[serde(skip)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    waiting: Vec<(GeometryHandle, Interest<RecordKey>)>,
    #[serde(skip)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    ended: Vec<(GeometryHandle, Interest<RecordKey>)>,
    #[serde(skip)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    skipped: Vec<GeometryHandle>,
}

fn lod_record(input: &[u8]) -> RecordKey {
    record_key(
        &engine_lodbuild::SOURCE_DIGEST,
        MESH_LODS_NAME,
        &content_hash(input).to_le_bytes(),
    )
}

fn idle(world: &World, awaiting: &[GeometryHandle], building: bool) -> bool {
    let (Some(levels), Some(assets)) = (
        world.get_resource::<MeshLevels>(),
        world.get_resource::<AssetLibrary>(),
    ) else {
        return true;
    };
    let gone = |handle: GeometryHandle| assets.mesh(handle).is_none();
    awaiting.is_empty()
        && levels.waiting.is_empty()
        && (!building || levels.skipped.is_empty())
        && !levels.ended.iter().any(|&(handle, _)| gone(handle))
        && !levels.skipped.iter().copied().any(gone)
}

pub(crate) fn settle(world: &mut World) {
    let mut awaiting = world
        .resource_mut::<AssetLibrary>()
        .bypass_change_detection()
        .take_awaiting_lods();
    let building = world
        .get_resource::<Detail>()
        .is_none_or(|detail| detail.chosen_error().is_some());
    if idle(world, &awaiting, building) {
        return;
    }
    let mut orders = Vec::new();
    world.resource_scope(|world, mut levels: Mut<MeshLevels>| {
        let levels = &mut *levels;
        if building {
            awaiting.append(&mut levels.skipped);
        }
        let mut inputs = Vec::new();
        if !awaiting.is_empty() {
            let mut assets = world.resource_mut::<AssetLibrary>();
            for handle in awaiting {
                let Some(mesh) = assets.mesh(handle) else {
                    continue;
                };
                if !building {
                    let _ = assets.settle_lods(handle, None);
                    levels.skipped.push(handle);
                    continue;
                }
                match lod_input(mesh) {
                    Some(input) => inputs.push((handle, input)),
                    None => {
                        let _ = assets.settle_lods(handle, None);
                    }
                }
            }
        }
        let records = world.resource::<Records>();
        for (handle, input) in inputs {
            let key = lod_record(&input);
            let place = u32::try_from(levels.waiting.len()).unwrap_or(u32::MAX);
            levels
                .waiting
                .push((handle, records.requests.ask_later(key)));
            orders.push(JobOrder::Start {
                key: key.hash(),
                kind: JobRef::from(&MESH_LODS),
                input,
                priority: job_priority(VIEW_JOBS, place),
                admitted: true,
                cause: engine_profile::carry(),
            });
        }
        let mut landed = Vec::new();
        for (handle, interest) in std::mem::take(&mut levels.waiting) {
            match records.request(&interest) {
                Some(Request::Ready(bytes)) => {
                    landed.push((handle, MeshLods::from_bytes(bytes).ok().map(Arc::new)));
                }
                Some(Request::Failed(_) | Request::Missing) => {
                    landed.push((handle, None));
                    levels.ended.push((handle, interest));
                }
                Some(Request::Asked(_)) | None => levels.waiting.push((handle, interest)),
            }
        }
        if !landed.is_empty() {
            let mut assets = world.resource_mut::<AssetLibrary>();
            for (handle, lods) in landed {
                if lods.is_some() && assets.settle_lods(handle, lods).is_ok() {
                    continue;
                }
                if assets.mesh(handle).is_some() && assets.mesh_lods(handle).is_none() {
                    let _ = assets.settle_lods(handle, None);
                }
            }
        }
        let assets = world.resource::<AssetLibrary>();
        levels
            .ended
            .retain(|&(handle, _)| assets.mesh(handle).is_some());
        levels
            .skipped
            .retain(|&handle| assets.mesh(handle).is_some());
    });
    if !orders.is_empty() {
        let batch = JobBatch { owner: 0, run: 0 };
        world
            .get_resource_or_init::<JobQueue>()
            .orders
            .extend(orders.into_iter().map(|order| (batch, order)));
    }
}
