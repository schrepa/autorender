//! Baking a terrain's layers: one record, every bake of its stack.

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
}

/// Bakes run before any tile loads: `state`'s layers (`authored`, until
/// baked) take every bake as one record, which the host's packs and store
/// hand back or a job makes, asked for once `slots` admits it and again a
/// while after it failed. Whether a bake is not done yet, which no tile
/// loads behind.
pub(super) fn bake(
    state: &mut TerrainState,
    authored: &Arc<Layers>,
    slots: &mut JobBudget<'_>,
    frame: u64,
) -> bool {
    let layers = state.baked.clone().unwrap_or_else(|| authored.clone());
    if layers.unbaked().is_none() {
        return false;
    }
    let Some(id) = state.bake.job else {
        if state.bake.failed.is_some_and(|at| frame < at + RETRY_AFTER) {
            return true;
        }
        let priority = job_priority(URGENT_JOBS, 0);
        // Admitted as every owner's jobs are; tiles wait
        // for the bake meanwhile.
        if !slots.take(priority) {
            return true;
        }
        let input = BakeJob {
            layers: (*layers).clone(),
            pyramid: state.terrain.pyramid,
        }
        .input();
        // As its layers can hold whole heightmaps, its identity is
        // its input's hash.
        let identity = content_hash(&input).to_le_bytes();
        let record = RecordKey::new(&engine_terrain::SOURCE, BakeJob::KIND.name(), &identity);
        state.bake.key = Some(record);
        state.bake.job = Some(slots.jobs.submit_record(
            BakeJob::KIND.name(),
            input,
            priority,
            record,
        ));
        return true;
    };
    let Some(output) = slots.jobs.take(id) else {
        return true;
    };
    state.bake.job = None;
    let baked = match job_output(
        BakeJob::KIND.name(),
        output,
        engine_terrain::jobs::bake_output,
    ) {
        Ok(baked) => baked,
        Err(error) => {
            state.failures.record(frame, error);
            state.bake.failed = Some(frame);
            return true;
        }
    };
    for layer in baked.layers() {
        match layer {
            Layer::Hydrology(HydrologyStage {
                baked: Some(found), ..
            }) => state.hydrology = HydrologyState::baked(found.stats),
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
    false
}
