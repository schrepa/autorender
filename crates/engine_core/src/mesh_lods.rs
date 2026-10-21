//! Meshes' levels of detail, which the cook makes offline and runtimes
//! only load: every mesh the library takes without levels of its own asks
//! hosts for its levels' record, keyed by the code making them
//! (`engine_lodbuild`, which no runtime links) and the mesh's content, and
//! draws once the answer is here, with them, or with none where no pack
//! or store holds them. A cooking session asks for each as the job making
//! it, which the cook's host runs where its packs lack the record.

use bevy_ecs::prelude::*;
use engine_assets::{AssetLibrary, LOD_RECORD, MeshHandle, MeshLods, lod_input};

use crate::{Cooking, JobId, Jobs, RecordKey, VIEW_JOBS, content_hash, job_priority};

// `SOURCE` of the code making the levels' records, which keys them.
mod maker {
    include!(concat!(env!("OUT_DIR"), "/engine_lodbuild_source.rs"));
}

/// The levels' records asked for: each mesh's, and the lookup or job under
/// way.
#[derive(Resource, Default)]
pub(crate) struct MeshLevels {
    asked: Vec<(MeshHandle, JobId)>,
}

/// The key of the levels' record of the mesh whose level job's input is
/// `input` ([`lod_input`]).
fn lod_record(input: &[u8]) -> RecordKey {
    RecordKey::new(
        &maker::SOURCE,
        LOD_RECORD,
        &content_hash(input).to_le_bytes(),
    )
}

/// Each rendered frame, as the world follows the camera: asks for the
/// levels of the meshes the library took since, and gives meshes those that
/// came: decoded where they suit their mesh, none otherwise.
pub(crate) fn settle(world: &mut World) {
    let cooking = world.contains_resource::<Cooking>();
    let awaiting = world.resource_mut::<AssetLibrary>().take_awaiting_lods();
    world.resource_scope(|world, mut levels: Mut<MeshLevels>| {
        world.resource_scope(|world, mut jobs: Mut<Jobs>| {
            let mut assets = world.resource_mut::<AssetLibrary>();
            for handle in awaiting {
                let Some(input) = assets.mesh(handle).and_then(|mesh| lod_input(mesh)) else {
                    let _ = assets.settle_lods(handle, None);
                    continue;
                };
                let key = lod_record(&input);
                let priority = job_priority(VIEW_JOBS, levels.asked.len() as u32);
                let id = if cooking {
                    jobs.submit_record(LOD_RECORD, input, priority, key)
                } else {
                    jobs.fetch(key, priority)
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
                // Levels that do not suit the mesh leave it without.
                if assets.settle_lods(handle, lods).is_err() {
                    let _ = assets.settle_lods(handle, None);
                }
                false
            });
        });
        let left = levels.asked.len() as u64;
        if let Some(mut cooking) = world.get_resource_mut::<Cooking>() {
            cooking.left("mesh_lods", left);
        }
    });
}
