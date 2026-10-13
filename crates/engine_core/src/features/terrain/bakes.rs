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

/// Bakes run before any tile loads, each reading the layers below it, in
/// stack order: `layers`' first bake not done advances on a worker where
/// `slots` has workers, unless one failed there, or here within `work`'s
/// time from slices of `budget`. Whether a bake was not done, which no tile
/// loads behind.
pub(super) fn bake(
    state: &mut TerrainState,
    layers: &Arc<Layers>,
    work: &Work,
    budget: &mut u64,
    mut slots: Option<&mut JobBudget<'_>>,
) -> bool {
    let Some(unbaked) = layers.unbaked() else {
        return false;
    };
    let pyramid = state.terrain.pyramid;
    if let Some(slots) = slots.as_deref_mut()
        && slots.jobs.workers() > 0
        && !state.bake_inline
    {
        bake_on_worker(state, slots, layers, &pyramid);
        return true;
    }
    // A bake handed to workers that stopped runs here.
    if let Some(id) = state.bake_job.take()
        && let Some(slots) = slots
    {
        slots.jobs.cancel(id);
    }
    let bake = state
        .bake
        .get_or_insert_with(|| TerrainBake::new(layers, unbaked.clone(), &pyramid));
    let _profile = engine_profile::span("cpu", "terrain_bake");
    let baked = loop {
        if let Some(baked) = bake.advance(&pyramid, budget) {
            break Some(baked);
        }
        if !work.refill(budget) {
            break None;
        }
    };
    if let Some(baked) = baked {
        match &baked {
            Baked::Hydrology(found) => state.hydrology = Some(found.stats),
            Baked::Erosion(found) => state.erosion = Some(found.stats),
            Baked::Path(_) => {}
        }
        let index = unbaked.0;
        finish_bake(state, Arc::new(layers.with_baked(index, baked)));
    }
    true
}

/// Bakes `layers` over `pyramid` on a worker: hands it every
/// bake at once, once `slots` has a slot free, then makes what it baked the
/// layers tiles generate from. A bake that failed there runs in place
/// instead.
pub(super) fn bake_on_worker(
    state: &mut TerrainState,
    slots: &mut JobBudget<'_>,
    layers: &Layers,
    pyramid: &Pyramid,
) {
    let Some(id) = state.bake_job else {
        let priority = job_priority(URGENT_JOBS, 0);
        // Admitted as every owner's jobs are; tiles wait
        // for the bake meanwhile.
        if !slots.take(priority) {
            return;
        }
        let input = BakeJob {
            layers: layers.clone(),
            pyramid: *pyramid,
        }
        .input();
        // Hosts may keep the baked layers, by the input's
        // hash.
        let identity = content_hash(&input).to_le_bytes().to_vec();
        state.bake_job = Some(slots.jobs.submit_cacheable(
            BakeJob::KIND.name(),
            input,
            priority,
            identity,
        ));
        return;
    };
    let Some(output) = slots.jobs.take(id) else {
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
