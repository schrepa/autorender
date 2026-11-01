use super::*;

pub(super) fn finish_bake(state: &mut TerrainState, baked: Arc<Layers>) {
    if baked.unbaked().is_none() {
        state.bridges = baked.bridges().into();
        state.standing += 1;
    }
    state.baked = Some(baked);
}

pub(super) fn bake(
    state: &mut TerrainState,
    authored: &Arc<Layers>,
    asking: &mut Asking<'_, '_, '_>,
    frame: u64,
) -> bool {
    let layers = state.bake_source(authored, asking.drawn);
    if layers.unbaked().is_none() {
        return false;
    }
    let Some(ticket) = state.bake.job.as_ref() else {
        if state.bake.failed.is_some_and(|at| frame < at + RETRY_AFTER) {
            return true;
        }
        let priority = job_priority(URGENT_JOBS, 0);
        let input = BakeJob {
            layers: (*layers).clone(),
            pyramid: state.terrain.pyramid,
        }
        .input();
        state.bake.drawn = Some(state.terrain.rivers.clone());
        state.bake.job = Some(asking.jobs.fetch(&BAKE, &input, priority));
        return true;
    };
    let Some(output) = asking.take(ticket) else {
        return true;
    };
    if let Some(ticket) = state.bake.job.take() {
        asking.jobs.cancel(ticket);
    }
    let drawn = state
        .bake
        .drawn
        .take()
        .unwrap_or_else(|| state.terrain.rivers.clone());
    let baked = match job_output("terrain::bake", output, crate::jobs::bake_output) {
        Ok(baked) => baked,
        Err(error) => {
            state.failures.record(frame, error);
            state.bake.failed = Some(frame);
            return true;
        }
    };
    if let Some(water) = baked.hydrology() {
        state.hydrology = HydrologyState::baked(water);
    }
    let before = state.baked.clone();
    finish_bake(state, Arc::new(baked));
    state.redrawn(drawn, before.as_deref());
    false
}
