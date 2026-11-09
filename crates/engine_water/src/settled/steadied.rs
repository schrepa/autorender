use std::sync::Arc;

use engine_core::{Error, ErrorCode, Interest, RecordKey, Request, VIEW_JOBS, job_priority};
use engine_heightfield::tile::TILE_STRIDE;
use engine_heightfield::{Pyramid, TILE_QUADS};
use glam::Vec2;

use super::{Follow, Found, Settled};
use crate::TileKey;
use crate::drawn::draw_input;
use crate::jobs::drawn::is_dry;
use crate::jobs::{DRAWN, STEADIED, SteadiedJob};
use crate::state::{Base, Drawing, WaterTerrain};

const TILE_ARRAY_BYTES: u64 = (TILE_STRIDE as u64) * (TILE_STRIDE as u64) * 4;

pub(super) struct Steadying {
    tile: TileKey,
    pub(super) job: Interest<RecordKey>,
    reads: Vec<Interest<RecordKey>>,
}

pub(super) struct Watering {
    reads: Vec<(TileKey, RecordKey)>,
    pub(super) water: Interest<RecordKey>,
}

enum Wants {
    Here(Vec<(TileKey, RecordKey)>),
    Coming,
    Nothing,
}

pub(super) fn water_tiles(
    terrain: &mut WaterTerrain,
    cx: &mut Follow<'_, '_, '_>,
    (watered, near): (&[TileKey], &[TileKey]),
    (eye, frame): (Vec2, u64),
) {
    let took = take_waters(terrain, cx, frame);
    let asks = std::mem::take(&mut terrain.settled.water_again);
    if asks {
        ask_waters(terrain, cx, (watered, near), eye);
    }
    let gives = std::mem::take(&mut terrain.settled.give_again);
    if took || asks || gives {
        terrain.settled.give_again = give_waters(terrain, cx);
    }
}

fn take_waters(terrain: &mut WaterTerrain, cx: &mut Follow<'_, '_, '_>, frame: u64) -> bool {
    let mut taken = false;
    let mut flying = Vec::new();
    for flight in std::mem::take(&mut terrain.settled.steadying) {
        if taken && cx.budget.expired() {
            flying.push(flight);
            continue;
        }
        let output = match cx.records.request(&flight.job) {
            None | Some(Request::Asked(_)) => {
                flying.push(flight);
                continue;
            }
            Some(Request::Ready(bytes)) => Ok(Arc::clone(bytes)),
            Some(Request::Failed(error)) => Err(error.to_string()),
            Some(Request::Missing) => Err(format!("{}: no record", STEADIED.name())),
        };
        taken = true;
        let Steadying { tile, job, reads } = flight;
        drop(reads);
        let water = *job.key();
        drop(job);
        let made = output.and_then(|bytes| {
            is_dry(&bytes).map(|_| bytes).map_err(|error| {
                format!("{}: its output does not decode: {error}", STEADIED.name())
            })
        });
        let resident = terrain.resident(&tile).is_some();
        let settled = &mut terrain.settled;
        let Some(ticket) = settled.waters.ticket(&water) else {
            continue;
        };
        match made {
            Ok(bytes) => {
                settled
                    .waters
                    .ready(&water, ticket, resident.then_some(bytes));
            }
            Err(error) => {
                let failed = Error::new(
                    ErrorCode::InvalidRecord,
                    format!("$.jobs.{}", STEADIED.name()),
                )
                .got(error.as_str());
                settled.waters.fail(&water, ticket, failed);
                cx.failures.record(frame, error);
            }
        }
    }
    flying.append(&mut terrain.settled.steadying);
    terrain.settled.steadying = flying;
    taken
}

fn watering(terrain: &WaterTerrain, (watered, near): (&[TileKey], &[TileKey])) -> Vec<TileKey> {
    let mut tiles = terrain.resident_keys(|key| {
        let Some(held) = terrain.resident(key) else {
            return false;
        };
        let held_back = held.watered.is_some();
        !held_back
            && if key.level == 0 {
                near.binary_search(key).is_ok()
            } else {
                held.water.is_some()
            }
    });
    if terrain.settled.cooking {
        let pyramid = terrain.pyramid;
        tiles.extend_from_slice(watered);
        for level in 1..pyramid.levels() {
            let [columns, rows] = pyramid.tiles(level);
            for key in watered {
                for dz in -1..=1_i32 {
                    for dx in -1..=1_i32 {
                        let (x, z) = (i32::from(key.x) + dx, i32::from(key.z) + dz);
                        if x < 0 || z < 0 {
                            continue;
                        }
                        let (x, z) = ((x as u32) >> level, (z as u32) >> level);
                        if x < columns && z < rows {
                            tiles.push(TileKey::new(level as u8, x as u16, z as u16));
                        }
                    }
                }
            }
        }
        tiles.sort_unstable();
        tiles.dedup();
    }
    tiles
}

fn wants(
    settled: &Settled,
    (pyramid, key): (&Pyramid, TileKey),
    (watered, near): (&[TileKey], &[TileKey]),
    settles: bool,
) -> Wants {
    let made = |at: TileKey| (settled.cooking || near.binary_search(&at).is_ok(), settles);
    let mut reads = Vec::new();
    let [columns, rows] = pyramid.tiles(0);
    if key.level == 0 {
        match settled.found(key, true, made(key)) {
            Found::Here(record) => reads.push((key, record)),
            Found::Coming => return Wants::Coming,
            Found::Never => return Wants::Nothing,
        }
        for dz in -1..=1_i32 {
            for dx in -1..=1_i32 {
                let (x, z) = (i32::from(key.x) + dx, i32::from(key.z) + dz);
                if (dx, dz) == (0, 0) || x < 0 || z < 0 || x >= columns as i32 || z >= rows as i32 {
                    continue;
                }
                let at = TileKey::new(0, x as u16, z as u16);
                if watered.binary_search(&at).is_err() {
                    continue;
                }
                match settled.found(at, true, made(at)) {
                    Found::Here(record) => reads.push((at, record)),
                    Found::Coming => return Wants::Coming,
                    Found::Never => {}
                }
            }
        }
    } else {
        let up = u32::from(key.level);
        let around = (2_u32 << up).div_ceil(TILE_QUADS).max(1);
        let span = |at: u16, count: u32| {
            let first = (u32::from(at) << up).saturating_sub(around);
            let last = (((u32::from(at) + 1) << up) + around - 1).min(count - 1);
            (first, last)
        };
        let (x0, x1) = span(key.x, columns);
        let (z0, z1) = span(key.z, rows);
        let from = watered.partition_point(|at| u32::from(at.x) < x0);
        for &at in watered[from..]
            .iter()
            .take_while(|at| u32::from(at.x) <= x1)
        {
            if !(z0..=z1).contains(&u32::from(at.z)) {
                continue;
            }
            match settled.found(at, false, made(at)) {
                Found::Here(record) => reads.push((at, record)),
                Found::Coming => return Wants::Coming,
                Found::Never => {}
            }
        }
    }
    if reads.is_empty() {
        return Wants::Nothing;
    }
    reads.sort_unstable_by_key(|&(at, _)| at);
    Wants::Here(reads)
}

fn ask_waters(
    terrain: &mut WaterTerrain,
    cx: &mut Follow<'_, '_, '_>,
    (watered, near): (&[TileKey], &[TileKey]),
    eye: Vec2,
) {
    let Some(cell) = terrain.settled.cell() else {
        return;
    };
    let pyramid = terrain.pyramid;
    let tiles = watering(terrain, (watered, near));
    terrain
        .settled
        .watering
        .retain(|key, _| tiles.binary_search(key).is_ok());
    let size = pyramid.tile_size(0);
    let mut wanted: Vec<(u32, TileKey)> = tiles
        .iter()
        .map(|&key| {
            let span = pyramid.tile_size(u32::from(key.level));
            let middle = pyramid.tile_origin(key) + Vec2::splat(0.5 * span);
            ((middle.distance(eye) / size) as u32, key)
        })
        .collect();
    wanted.sort_unstable();
    let mut waiting = 0;
    for (place, key) in wanted {
        let reads = match wants(
            &terrain.settled,
            (&pyramid, key),
            (watered, near),
            cx.settles,
        ) {
            Wants::Here(reads) => reads,
            Wants::Coming => {
                waiting += 1;
                continue;
            }
            Wants::Nothing => continue,
        };
        let asked = terrain
            .settled
            .watering
            .get(&key)
            .is_some_and(|watering| watering.reads == reads);
        if asked {
            continue;
        }
        let Some(tile) = terrain.tile_job(key) else {
            continue;
        };
        let shared: Option<Vec<Interest<RecordKey>>> = reads
            .iter()
            .map(|&(_, shared)| cx.shares.keep(shared))
            .collect();
        let Some(shared) = shared else {
            continue;
        };
        let input = SteadiedJob {
            tile,
            cell,
            settled: reads
                .iter()
                .map(|&(at, shared)| (at, shared.hash()))
                .collect(),
        }
        .input();
        let record = RecordKey::of(&STEADIED, &input);
        let settled = &mut terrain.settled;
        let fresh = settled.waters.get(&record).is_none();
        let water = settled.waters.ask(record);
        if fresh {
            let job = cx
                .jobs
                .start(&STEADIED, input, job_priority(VIEW_JOBS, place));
            settled.steadying.push(Steadying {
                tile: key,
                job,
                reads: shared,
            });
        }
        settled.watering.insert(key, Watering { reads, water });
    }
    terrain.settled.waters_waiting = waiting;
}

fn give_waters(terrain: &mut WaterTerrain, cx: &mut Follow<'_, '_, '_>) -> bool {
    let settled = &terrain.settled;
    let giving: Vec<(TileKey, Arc<[u8]>, RecordKey)> = settled
        .watering
        .iter()
        .filter_map(|(&key, watering)| {
            let record = *watering.water.key();
            let Some(Request::Ready(Some(bytes))) = settled.waters.get(&record) else {
                return None;
            };
            let bytes = Arc::clone(bytes);
            let held = terrain.resident(&key)?;
            let coming =
                matches!(&held.base, Base::Redrawing(_, Drawing::Given(given)) if *given == record);
            let holds = coming || held.given == Some(record);
            (held.face.current && held.watered.is_none() && !holds).then_some((key, bytes, record))
        })
        .collect();
    let mut gave = false;
    for (key, bytes, record) in giving {
        if gave && cx.budget.expired() {
            return true;
        }
        if !cx.budget.spend("water", 2 * TILE_ARRAY_BYTES) {
            return true;
        }
        gave = true;
        let Some(held) = terrain.held.get_mut(&key) else {
            continue;
        };
        if matches!(is_dry(&bytes), Ok(true)) {
            held.water = None;
            held.drawn = None;
            held.watered = None;
            held.given = Some(record);
            held.base = Base::Made;
            terrain.touch(key);
        } else {
            let input = draw_input(held, &bytes);
            let job = cx.jobs.start(&DRAWN, input, job_priority(VIEW_JOBS, 0));
            held.base = Base::Redrawing(job, Drawing::Given(record));
        }
    }
    false
}
