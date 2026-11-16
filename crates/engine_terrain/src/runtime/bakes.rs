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
    let (baked, drawn, key) = match state.bake.take() {
        Some(mut bake) if bake.work.waiting() => {
            let landed = match bake.work.poll(asking.records) {
                Landed::Waiting => None,
                Landed::Ready(baked) => Some(Ok(Arc::clone(baked))),
                Landed::Failed(error) => Some(Err(format!("{}: {error}", BAKE.name()))),
            };
            match landed {
                None => {
                    state.bake = Some(bake);
                    return true;
                }
                Some(Ok(baked)) => {
                    let (drawn, key) = (Arc::clone(&bake.drawn), bake.key);
                    state.bake = Some(bake);
                    (baked, drawn, key)
                }
                Some(Err(error)) => {
                    state.failures.record(frame, error);
                    state.bake = Some(bake);
                    return true;
                }
            }
        }
        Some(bake)
            if bake.work.failure().is_some() && bake.drawn[..] == state.terrain.rivers[..] =>
        {
            state.bake = Some(bake);
            return true;
        }
        Some(_) | None => {
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
            state.bake = Some(TerrainBake {
                key: work.asked_key().copied(),
                work,
                drawn: state.terrain.rivers.clone(),
            });
            return true;
        }
    };
    if let Some(water) = baked.hydrology() {
        state.hydrology = HydrologyState::baked(water);
        state.hydrology_changed = true;
    }
    let before = state.baked.clone();
    state.finish_bake((baked, key), kinds);
    state.redrawn(drawn, before.as_deref());
    false
}
