//! Baking a terrain's layers, in place or on a worker.

use super::*;

/// Makes `baked` the layers `state`'s tiles generate from: once every bake
/// is done, with its roads' bridges (collision follows the
/// revision) and its rivers' waterfalls.
pub(super) fn finish_bake(state: &mut TerrainState, baked: Arc<Layers>) {
    if baked.unbaked().is_none() {
        state.bridges = baked
            .layers()
            .iter()
            .filter_map(|layer| match layer {
                Layer::Path(PathStage {
                    baked: Some(profile),
                    ..
                }) => Some(profile.bridges()),
                Layer::Noise(_)
                | Layer::Heightmap(_)
                | Layer::Stamp(_)
                | Layer::Hydrology(_)
                | Layer::Erosion(_)
                | Layer::Path(_) => None,
            })
            .flatten()
            .collect::<Vec<_>>()
            .into();
        state.falls = baked
            .hydrology()
            .map_or_else(|| Arc::from(Vec::new()), |water| water.falls.clone());
        state.standing += 1;
    }
    state.baked = Some(baked);
    state.bake = None;
}

/// Bakes `layers` over `pyramid` on a worker: hands it every
/// bake at once, then makes what it baked the layers tiles generate from. A
/// bake that failed there runs in place instead.
pub(super) fn bake_on_worker(
    state: &mut TerrainState,
    jobs: &mut Jobs,
    layers: &Layers,
    pyramid: &Pyramid,
) {
    let Some(id) = state.bake_job else {
        let input = BakeJob {
            layers: layers.clone(),
            pyramid: *pyramid,
        }
        .input();
        let priority = job_priority(URGENT_JOBS, 0);
        // Hosts may keep the baked layers, by the input's
        // hash.
        let identity = content_hash(&input).to_le_bytes().to_vec();
        state.bake_job = Some(jobs.submit_cacheable(BAKE_JOB, input, priority, identity));
        return;
    };
    let Some(output) = jobs.take(id) else {
        return;
    };
    state.bake_job = None;
    let Some(baked) = output
        .ok()
        .and_then(|bytes| engine_terrain::jobs::bake_output(&bytes).ok())
    else {
        state.bake_inline = true;
        return;
    };
    for layer in baked.layers() {
        match layer {
            Layer::Hydrology(HydrologyStage {
                baked: Some(found), ..
            }) => state.hydrology = Some(found.stats),
            Layer::Erosion(ErosionStage {
                baked: Some(found), ..
            }) => state.erosion = Some(found.stats),
            Layer::Noise(_)
            | Layer::Heightmap(_)
            | Layer::Stamp(_)
            | Layer::Hydrology(_)
            | Layer::Erosion(_)
            | Layer::Path(_) => {}
        }
    }
    finish_bake(state, Arc::new(baked));
}
