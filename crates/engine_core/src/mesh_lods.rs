use bevy_ecs::prelude::*;
use engine_assets::{AssetLibrary, MeshHandle, MeshLods, lod_input};

use crate::jobs::{MESH_LODS_NAME, record_key};
use crate::{
    CookMode, JobExchange, JobId, JobRef, MESH_LODS, RecordKey, VIEW_JOBS, content_hash,
    job_priority,
};

#[derive(Resource, Default)]
pub(crate) struct MeshLevels {
    asked: Vec<(MeshHandle, JobId)>,
}

impl MeshLevels {
    pub(crate) fn waiting(&self) -> usize {
        self.asked.len()
    }
}

fn lod_record(input: &[u8]) -> RecordKey {
    record_key(
        &crate::sources::LODBUILD,
        MESH_LODS_NAME,
        &content_hash(input).to_le_bytes(),
    )
}

pub(crate) fn settle(world: &mut World) {
    let cooking = world
        .get_resource::<CookMode>()
        .is_some_and(|mode| mode.cooking);
    let awaiting = world.resource_mut::<AssetLibrary>().take_awaiting_lods();
    world.resource_scope(|world, mut levels: Mut<MeshLevels>| {
        world.resource_scope(|world, mut jobs: Mut<JobExchange>| {
            let mut assets = world.resource_mut::<AssetLibrary>();
            for handle in awaiting {
                let Some(input) = assets.mesh(handle).and_then(|mesh| lod_input(mesh)) else {
                    let _ = assets.settle_lods(handle, None);
                    continue;
                };
                let key = lod_record(&input);
                let priority = job_priority(VIEW_JOBS, levels.asked.len() as u32);
                let id = if cooking {
                    jobs.submit_record(JobRef::from(&MESH_LODS), input, priority, key)
                } else {
                    jobs.fetch(Some(JobRef::from(&MESH_LODS)), key, priority)
                };
                levels.asked.push((handle, id));
            }
            levels.asked.retain(|&(handle, id)| {
                let Some(output) = jobs.take(id) else {
                    return true;
                };
                let lods = output
                    .ok()
                    .and_then(|bytes| MeshLods::from_bytes(&bytes).ok());
                if assets.settle_lods(handle, lods).is_err() {
                    let _ = assets.settle_lods(handle, None);
                }
                false
            });
        });
    });
}
