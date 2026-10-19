//! Tiles' water as their terrain's rivers run now. Tiles generate their
//! water from their bake's rivers; once what stands in the rivers holds
//! them back ([`super::flow`]), a resident tile whose water was made from
//! other rivers than run now is made again where the two differ
//! ([`Tile::rewatered`]), finest levels first within the frame's work, so
//! drawing, queries and walks read the rivers as they run.

use super::*;
use engine_terrain::{RiverFlow, Rivers};

/// Samples past their widest half rivers' water reaches at a level: rivers
/// widen to most of a sample and their water a sample and a half past
/// their edge.
const REACH_SAMPLES: f32 = 2.25;

/// Whether `a` and `b` are the same rivers.
fn same(a: &Watered, b: &Watered) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        _ => false,
    }
}

/// Makes the water of `state`'s resident tiles again where it was made
/// from other rivers than run now, within `work`'s time from slices of
/// `budget`: the tiles whose water it made again, and those left waiting.
pub(super) fn rewater(state: &mut TerrainState, work: &Work, budget: &mut u64) -> [u32; 2] {
    let Some(layers) = state.baked.clone() else {
        return [0, 0];
    };
    let Some(hydrology) = layers.hydrology() else {
        return [0, 0];
    };
    let baked = hydrology.rivers();
    // The bake's rivers unless something holds them back.
    let running: Watered = state
        .hydrology
        .flow
        .as_ref()
        .map(RiverFlow::current)
        .filter(|current| !Arc::ptr_eq(current, baked))
        .cloned();
    let store = &state.store;
    let mut stale =
        store.keys_where(|key| store.watered(key).is_some_and(|from| !same(from, &running)));
    if stale.is_empty() {
        return [0, 0];
    }
    stale.sort_unstable_by_key(|key| (key.level, key.x, key.z));
    let to: &Rivers = running.as_deref().unwrap_or(baked);
    let pyramid = state.terrain.pyramid;
    // Where the rivers running now differ from those each tile's water was
    // made from, by those rivers.
    let mut regions: Vec<(Watered, Vec<(Vec2, Vec2)>)> = Vec::new();
    let [mut made, mut waiting] = [0, 0];
    for key in stale {
        let Some(from) = state.store.watered(&key).cloned() else {
            continue;
        };
        let seen = match regions.iter().position(|(seen, _)| same(seen, &from)) {
            Some(seen) => seen,
            None => {
                let differs = to.differs_from(from.as_deref().unwrap_or(baked));
                regions.push((from, differs));
                regions.len() - 1
            }
        };
        // Rivers running alike everywhere leave its water as it is.
        if regions[seen].1.is_empty() {
            state.store.rewater(&key, None, running.clone());
            continue;
        }
        if *budget == 0 && !work.refill(budget) {
            waiting += 1;
            continue;
        }
        *budget = budget.saturating_sub(u64::from(TILE_STRIDE * TILE_STRIDE));
        let reach = Vec2::splat(REACH_SAMPLES * pyramid.spacing(u32::from(key.level)) + 1.0);
        let reached: Vec<(Vec2, Vec2)> = regions[seen]
            .1
            .iter()
            .map(|&(low, high)| (low - reach, high + reach))
            .collect();
        let rewatered = state
            .store
            .get(&key)
            .and_then(|tile| tile.rewatered(&layers, &pyramid, to, &reached));
        made += u32::from(rewatered.is_some());
        state
            .store
            .rewater(&key, rewatered.map(Arc::new), running.clone());
    }
    [made, waiting]
}
