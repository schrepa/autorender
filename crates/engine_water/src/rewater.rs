use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{FrameBudget, Jobs, VIEW_JOBS, job_priority};
use engine_heightfield::TILE_SAMPLES;
use glam::Vec2;

use crate::RiverFlow;
use crate::drawn::{flood_input, start_drawing};
use crate::flow::Running;
use crate::jobs::REWATER;
use crate::state::{Base, Drawing, WaterState, WaterTerrain, Watered};
use crate::water_blocks::Blocks;

const REACH_SAMPLES: f32 = 2.25;

fn same(a: &Watered, b: &Watered) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a.name() == b.name(),
        (Some(_), None) | (None, Some(_)) => false,
    }
}

fn rewater_terrain(
    terrain: &mut WaterTerrain,
    jobs: &mut Jobs,
    blocks: &Blocks,
    budget: &FrameBudget,
) -> u32 {
    let layers = Arc::clone(&terrain.baked);
    let Some(hydrology) = layers.hydrology() else {
        return 0;
    };
    let baked = &hydrology.rivers;
    let running: Watered = terrain.flow.as_ref().and_then(RiverFlow::running).cloned();
    let mut stale = terrain.resident_keys(|key| {
        terrain
            .resident(key)
            .is_some_and(|held| !same(&held.watered, &running))
    });
    if stale.is_empty() {
        return 0;
    }
    stale.sort_unstable_by_key(|key| (key.level, key.x, key.z));
    let pyramid = terrain.pyramid;
    let entity = terrain.entity;
    let mut waiting = 0;
    stale.retain(|key| {
        let flying = terrain
            .held
            .get(key)
            .is_some_and(|held| matches!(held.base, Base::Redrawing(..)));
        waiting += u32::from(flying);
        !flying
    });
    let flow = running.as_ref().map_or(([0; 32], Vec::new()), |running| {
        (running.name(), running.encoded())
    });
    let mut regions: Vec<(Watered, Vec<(Vec2, Vec2)>)> = Vec::new();
    let mut sent = false;
    for key in stale {
        let Some(from) = terrain.resident(&key).map(|held| held.watered.clone()) else {
            continue;
        };
        let seen = match regions.iter().position(|(seen, _)| same(seen, &from)) {
            Some(seen) => seen,
            None => {
                let differs = Running::differs(baked, running.as_deref(), from.as_deref());
                regions.push((from, differs));
                regions.len() - 1
            }
        };
        let spacing = pyramid.spacing(u32::from(key.level));
        let reach = Vec2::splat(REACH_SAMPLES * spacing + 1.0);
        let [x, z] = pyramid.first_sample(key);
        let [dx, dz] = layers.offset_samples(spacing);
        let low = Vec2::new((x + dx - 1) as f32, (z + dz - 1) as f32) * spacing;
        let high = low + Vec2::splat((TILE_SAMPLES + 1) as f32 * spacing);
        let reached: Vec<[f32; 4]> = regions[seen]
            .1
            .iter()
            .map(|&(from, to)| (from - reach, to + reach))
            .filter(|(from, to)| from.cmple(high).all() && to.cmpge(low).all())
            .map(|(from, to)| [from.x, from.y, to.x, to.y])
            .collect();
        let Some(held) = terrain.held.get_mut(&key) else {
            continue;
        };
        if reached.is_empty() {
            let flipped = held.watered.is_some() != running.is_some();
            held.watered = running.clone();
            if flipped {
                terrain.touch(key);
            }
            continue;
        }
        let blocked = !blocks.caught_up(entity, key, held.shown()) || (sent && budget.expired());
        if blocked {
            waiting += 1;
            continue;
        }
        let Some(tile) = terrain.tile_job(key) else {
            continue;
        };
        let Some(held) = terrain.held.get_mut(&key) else {
            continue;
        };
        let input = flood_input(held, tile, (flow.0, &flow.1), reached);
        let job = start_drawing(jobs, &REWATER, input, job_priority(VIEW_JOBS, 0));
        held.base = Base::Redrawing(job, Drawing::Flood(running.clone()));
        sent = true;
    }
    waiting
}

pub(crate) fn rewater(
    budget: Res<FrameBudget>,
    mut jobs: Jobs,
    mut state: ResMut<WaterState>,
) {
    let state = &mut *state;
    let mut stale = 0;
    for terrain in &mut state.terrains {
        stale += rewater_terrain(terrain, &mut jobs, &state.blocks, &budget);
    }
    state.rewatered_tiles = 0;
    state.stale_water_tiles = stale;
}
