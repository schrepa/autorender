use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::WorkClock;
use engine_heightfield::tile::TILE_STRIDE;
use engine_places::ground::GroundWork;
use glam::Vec2;

use crate::state::{WaterState, WaterTerrain, Watered};
use crate::tile_water::rewatered;
use crate::work::Work;
use crate::{RiverFlow, Rivers};

const REACH_SAMPLES: f32 = 2.25;

fn same(a: &Watered, b: &Watered) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        _ => false,
    }
}

fn rewater_terrain(terrain: &mut WaterTerrain, work: &Work, budget: &mut u64) -> [u32; 2] {
    let layers = Arc::clone(&terrain.baked);
    let Some(hydrology) = layers.hydrology() else {
        return [0, 0];
    };
    let baked = &hydrology.rivers;
    let running: Watered = terrain
        .flow
        .as_ref()
        .map(RiverFlow::current)
        .filter(|current| !Arc::ptr_eq(current, baked))
        .cloned();
    let mut stale = terrain.resident_keys(|key| {
        terrain
            .resident(key)
            .is_some_and(|held| !same(&held.watered, &running))
    });
    if stale.is_empty() {
        return [0, 0];
    }
    stale.sort_unstable_by_key(|key| (key.level, key.x, key.z));
    let to: &Rivers = running.as_deref().unwrap_or(baked);
    let pyramid = terrain.pyramid;
    let mut regions: Vec<(Watered, Vec<(Vec2, Vec2)>)> = Vec::new();
    let [mut made, mut waiting] = [0, 0];
    for key in stale {
        let Some(from) = terrain.resident(&key).map(|held| held.watered.clone()) else {
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
        if regions[seen].1.is_empty() {
            if let Some(held) = terrain.held.get_mut(&key) {
                held.watered = running.clone();
            }
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
        let Some(held) = terrain.held.get_mut(&key) else {
            continue;
        };
        let rewatered = rewatered(
            &held.face,
            held.water.as_deref(),
            (&layers, &pyramid),
            to,
            &reached,
        );
        held.watered = running.clone();
        if let Some(water) = rewatered {
            made += 1;
            held.hold(Some(water));
            terrain.revision += 1;
        }
    }
    [made, waiting]
}

pub(crate) fn rewater(
    clock: Res<WorkClock>,
    ground: Res<GroundWork>,
    mut state: ResMut<WaterState>,
) {
    let work = Work::of(&clock, &ground);
    let mut budget = work.slice;
    let state = &mut *state;
    let [mut rewatered, mut stale] = [0, 0];
    for terrain in &mut state.terrains {
        let [made, waiting] = rewater_terrain(terrain, &work, &mut budget);
        rewatered += made;
        stale += waiting;
    }
    state.rewatered_tiles = rewatered;
    state.stale_water_tiles = stale;
}
