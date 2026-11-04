use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{JobExchange, JobId, JobRef, VIEW_JOBS, WorkClock, job_priority};
use engine_heightfield::{TILE_SAMPLES, TileKey};
use engine_places::ground::GroundWork;
use glam::Vec2;

use crate::RiverFlow;
use crate::drawn::flood_input;
use crate::flow::Running;
use crate::jobs::REWATER;
use crate::state::{Drawing, WaterState, WaterTerrain, Watered};
use crate::water_blocks::Blocks;
use crate::work::Work;

const REACH_SAMPLES: f32 = 2.25;

fn same(a: &Watered, b: &Watered) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (Some(_), None) | (None, Some(_)) => false,
    }
}

type Flights = Vec<(JobId, Entity, TileKey)>;

fn rewater_terrain(
    terrain: &mut WaterTerrain,
    (exchange, flights): (&mut JobExchange, &mut Flights),
    blocks: &Blocks,
    work: &Work,
) -> u32 {
    let touched = terrain.touched_all || !terrain.touched.is_empty();
    let due = std::mem::take(&mut terrain.flow_moved) || (terrain.flow.is_some() && touched);
    if !due {
        return 0;
    }
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
    let flow = running.as_ref().map_or(([0; 32], Vec::new()), |running| {
        (running.name(), running.encoded())
    });
    let mut regions: Vec<(Watered, Vec<(Vec2, Vec2)>)> = Vec::new();
    let (mut waiting, mut sent) = (0, false);
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
        let blocked = held.drawing.is_some()
            || !blocks.caught_up(entity, key, held.water.as_ref())
            || (sent && work.expired());
        if blocked {
            waiting += 1;
            terrain.flow_moved |= held.drawing.is_none();
            continue;
        }
        let Some(tile) = terrain.tile_job(key) else {
            continue;
        };
        let Some(held) = terrain.held.get_mut(&key) else {
            continue;
        };
        let input = flood_input(held, tile, (flow.0, &flow.1), reached);
        let job = exchange.submit(JobRef::from(&REWATER), input, job_priority(VIEW_JOBS, 0));
        held.drawing = Some((job, Drawing::Flood(running.clone())));
        flights.push((job, entity, key));
        sent = true;
    }
    waiting
}

pub(crate) fn rewater(
    mut exchange: ResMut<JobExchange>,
    clock: Res<WorkClock>,
    ground: Res<GroundWork>,
    mut state: ResMut<WaterState>,
) {
    let work = Work::until(&clock, &ground, state.work_end);
    let state = &mut *state;
    let mut stale = 0;
    for terrain in &mut state.terrains {
        stale += rewater_terrain(
            terrain,
            (&mut exchange, &mut state.flights),
            &state.blocks,
            &work,
        );
    }
    state.rewatered_tiles = 0;
    state.stale_water_tiles = stale;
}
