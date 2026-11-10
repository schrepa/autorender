use super::*;
use engine_core::Landed;

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
    let kinds = asking.kinds;
    let (baked, drawn) = match std::mem::take(&mut state.bake) {
        TerrainBake::Asked { mut work, drawn } => {
            let landed = match work.poll(asking.records) {
                Landed::Waiting => None,
                Landed::Ready(baked) => Some(Ok(Arc::clone(baked))),
                Landed::Failed(error) => Some(Err(format!("{}: {error}", BAKE.name()))),
            };
            match landed {
                None => {
                    state.bake = TerrainBake::Asked { work, drawn };
                    return true;
                }
                Some(Ok(baked)) => (baked, drawn),
                Some(Err(error)) => {
                    state.failures.record(frame, error);
                    state.bake = TerrainBake::Failed {
                        _failure: work,
                        drawn,
                    };
                    return true;
                }
            }
        }
        TerrainBake::Failed { _failure, drawn } if drawn[..] == state.terrain.rivers[..] => {
            state.bake = TerrainBake::Failed { _failure, drawn };
            return true;
        }
        TerrainBake::Unasked | TerrainBake::Failed { .. } | TerrainBake::Made => {
            let priority = job_priority(URGENT_JOBS, 0);
            let input = BakeJob {
                layers: (*layers).clone(),
                pyramid: state.terrain.pyramid,
            }
            .input();
            state.changed = true;
            let listed = kinds.to_vec();
            let work = asking.jobs.fetch(&BAKE, &input, priority, move |bytes| {
                crate::jobs::bake_output(&bytes, &listed)
                    .map(Arc::new)
                    .map_err(|error| BAKE.undecoded(error))
            });
            state.bake = TerrainBake::Asked {
                work,
                drawn: state.terrain.rivers.clone(),
            };
            return true;
        }
    };
    state.bake = TerrainBake::Made;
    if let Some(water) = baked.hydrology() {
        state.hydrology = HydrologyState::baked(water);
        state.hydrology_changed = true;
    }
    let before = state.baked.clone();
    state.finish_bake(baked, kinds);
    state.redrawn(drawn, before.as_deref());
    false
}
