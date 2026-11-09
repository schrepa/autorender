use super::*;

pub(super) fn bake(
    state: &mut TerrainState,
    authored: &Arc<Layers>,
    asking: &mut Asking<'_, '_, '_>,
    frame: u64,
) -> bool {
    let layers = state.bake_source(authored, asking.kinds);
    if layers.unbaked().is_none() {
        return false;
    }
    let arrived = match std::mem::take(&mut state.bake) {
        TerrainBake::Asked { interest, drawn } => match asking.take(&interest) {
            Some(output) => Some((output, interest, drawn)),
            None => {
                state.bake = TerrainBake::Asked { interest, drawn };
                return true;
            }
        },
        TerrainBake::Failed { interest, drawn } if drawn[..] == state.terrain.rivers[..] => {
            state.bake = TerrainBake::Failed { interest, drawn };
            return true;
        }
        TerrainBake::Unasked | TerrainBake::Failed { .. } | TerrainBake::Made => None,
    };
    let Some((output, interest, drawn)) = arrived else {
        let priority = job_priority(URGENT_JOBS, 0);
        let input = BakeJob {
            layers: (*layers).clone(),
            pyramid: state.terrain.pyramid,
        }
        .input();
        state.changed = true;
        state.bake = TerrainBake::Asked {
            interest: asking.jobs.fetch(&BAKE, &input, priority),
            drawn: state.terrain.rivers.clone(),
        };
        return true;
    };
    let kinds = asking.kinds;
    let baked = match job_output(BAKE.name(), output, |bytes| {
        crate::jobs::bake_output(bytes, kinds)
    }) {
        Ok(baked) => baked,
        Err(error) => {
            state.failures.record(frame, error);
            state.bake = TerrainBake::Failed { interest, drawn };
            return true;
        }
    };
    state.bake = TerrainBake::Made;
    if let Some(water) = baked.hydrology() {
        state.hydrology = HydrologyState::baked(water);
        state.hydrology_changed = true;
    }
    let before = state.baked.clone();
    state.finish_bake(Arc::new(baked), kinds);
    state.redrawn(drawn, before.as_deref());
    false
}
