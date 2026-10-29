use super::*;

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
                | Layer::Rivers(_)
                | Layer::Hydrology(_)
                | Layer::Erosion(_)
                | Layer::Path(_)
                | Layer::Steps(_)
                | Layer::Hummocks(_) => None,
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

pub(super) fn bake(
    state: &mut TerrainState,
    authored: &Arc<Layers>,
    slots: &mut JobBudget<'_>,
    frame: u64,
) -> bool {
    let layers = state.bake_source(authored);
    if layers.unbaked().is_none() {
        return false;
    }
    let Some(id) = state.bake.job else {
        if state.bake.failed.is_some_and(|at| frame < at + RETRY_AFTER) {
            return true;
        }
        let priority = job_priority(URGENT_JOBS, 0);
        if !slots.take(priority) {
            return true;
        }
        let input = BakeJob {
            layers: (*layers).clone(),
            pyramid: state.terrain.pyramid,
        }
        .input();
        let identity = content_hash(&input).to_le_bytes();
        let record = RecordKey::new(&crate::sources::BAKES, BakeJob::KIND.name(), &identity);
        state.bake.key = Some(record);
        state.bake.drawn = Some(state.terrain.rivers.clone());
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
    let drawn = state
        .bake
        .drawn
        .take()
        .unwrap_or_else(|| state.terrain.rivers.clone());
    let baked = match job_output(BakeJob::KIND.name(), output, crate::jobs::bake_output) {
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
            | Layer::Rivers(_)
            | Layer::Hydrology(_)
            | Layer::Erosion(_)
            | Layer::Path(_)
            | Layer::Steps(_)
            | Layer::Hummocks(_) => {}
        }
    }
    let before = state.baked.clone();
    finish_bake(state, Arc::new(baked));
    state.redrawn(drawn, before.as_deref());
    false
}
