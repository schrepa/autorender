use std::collections::BTreeSet;
use std::sync::Arc;

use engine_core::{
    Error, ErrorCode, RecordKey, Request, Taken, VIEW_JOBS, job_priority, record_missing,
};

use super::{
    Follow, Landed, Making, Settle, Settled, arrive, both, claim, end, record_input, water_part,
};
use crate::jobs::{PARTS, SETTLED, WaterJob};
use crate::state::WaterTerrain;
use crate::water_settle::SettleAsk;
use crate::{SettleRegion, SettleStart, SteadyCell, TileKey};

pub(super) fn cook_regions(terrain: &WaterTerrain) -> Vec<(SettleRegion, bool, u32)> {
    let pyramid = terrain.pyramid;
    let settled = &terrain.settled;
    let Some(reach) = settled.held_reach() else {
        return Vec::new();
    };
    let cell = reach.cell;
    let made: BTreeSet<TileKey> = settled
        .making
        .iter()
        .flat_map(|making| making.settles.iter().map(|settle| settle.tile))
        .collect();
    let state = |key: TileKey| {
        settled.tiles.get(&key).map_or((false, false), |tile| {
            (settled.asked(tile), settled.failed(tile).is_some())
        })
    };
    let mut ready = Vec::new();
    for region in reach.regions.iter() {
        let (rectangle, tiles) = (region.tiles, &region.keys);
        let fetched = tiles.iter().all(|&key| {
            let (asked, failed) = state(key);
            let fetching = asked && !made.contains(&key);
            !fetching && (settled.done(key, false) || settled.missing(key) || failed || asked)
        });
        let kept: Vec<TileKey> = tiles
            .iter()
            .copied()
            .filter(|&key| {
                let (asked, failed) = state(key);
                settled.missing(key) && !asked && !failed
            })
            .collect();
        if fetched && !kept.is_empty() {
            ready.push((
                SettleRegion::over(&pyramid, rectangle, kept, cell),
                false,
                0,
            ));
        }
    }
    ready
}

pub(super) fn start_making(
    terrain: &mut WaterTerrain,
    cx: &mut Follow<'_, '_, '_>,
    (recipe, whole): ([u8; 32], &[TileKey]),
    region: SettleRegion,
    (alone, warm, place): (bool, bool, u32),
) {
    let Some(cell) = terrain.settled.cell() else {
        return;
    };
    let mut settles = Vec::with_capacity(region.kept.len());
    for &key in &region.kept {
        let single = terrain.settled.single(key);
        let alone = alone || single;
        let Some(record) = record_input(terrain, key, alone) else {
            return;
        };
        let at = usize::from(alone != single);
        let chained = terrain
            .settled
            .tiles
            .get(&key)
            .map_or(0, |tile| tile.asking.len());
        if at > chained {
            return;
        }
        settles.push((
            Settle {
                tile: key,
                alone,
                record,
            },
            at,
        ));
    }
    let reached = region.reached(&terrain.pyramid);
    let jobs: Vec<WaterJob> = reached
        .iter()
        .map(|&key| water_part(terrain, recipe, key, cell))
        .collect();
    let Settled {
        records,
        tiles,
        parts,
        part_jobs,
        making,
        ..
    } = &mut terrain.settled;
    for (settle, at) in &settles {
        let wide = whole.binary_search(&settle.tile).is_ok();
        let tile = tiles.entry(settle.tile).or_default();
        claim(records, tile, (*at, settle.record.key), wide);
        for wanted in both(settle.record.key) {
            if matches!(records.get(&wanted), Some(Request::Missing)) {
                records.ask_again(&wanted);
            }
        }
    }
    let priority = job_priority(VIEW_JOBS, place);
    let mut wanted = Vec::with_capacity(reached.len());
    for (&key, job) in reached.iter().zip(jobs) {
        let input = job.input();
        let part = RecordKey::of(&PARTS, &input);
        let fresh = parts.get(&part).is_none();
        wanted.push(parts.ask(part));
        if fresh {
            let work = cx.jobs.start(&PARTS, input, priority, |bytes| {
                crate::jobs::water_output(&bytes).map_err(|error| PARTS.undecoded(error))
            });
            part_jobs.push((key, part, work));
        }
    }
    making.push(Making {
        region,
        settles: settles.into_iter().map(|(settle, _)| settle).collect(),
        warm,
        parts: wanted,
        settling: false,
    });
}

pub(super) fn take_settles(
    terrain: &mut WaterTerrain,
    cx: &mut Follow<'_, '_, '_>,
    (kept, frame): (&[TileKey], u64),
    landed: &mut Vec<Landed>,
) {
    let mut waiting = Vec::new();
    for making in std::mem::take(&mut terrain.settled.making) {
        if !making.settling {
            waiting.push(making);
            continue;
        }
        let outputs: Vec<_> = making
            .settles
            .iter()
            .map(|settle| cx.queue.made(&settle.record.bytes))
            .collect();
        if outputs.iter().all(Option::is_none) {
            waiting.push(making);
            continue;
        }
        let failed = |got: &str| {
            Error::new(
                ErrorCode::InvalidRecord,
                format!("$.jobs.{}", SETTLED.name()),
            )
            .got(got)
        };
        for (settle, output) in making.settles.iter().zip(outputs) {
            let output = match output {
                Some(Ok(bytes)) => Ok(bytes),
                Some(Err(message)) => Err(Some(failed(&message))),
                None => Err(Some(failed("its square settled without it"))),
            };
            let at = (settle.tile, settle.alone, settle.record.key);
            landed.extend(arrive(terrain, cx, at, output, (kept, frame)));
        }
    }
    waiting.append(&mut terrain.settled.making);
    terrain.settled.making = waiting;
}

pub(super) fn take_parts(terrain: &mut WaterTerrain, cx: &mut Follow<'_, '_, '_>, frame: u64) {
    let cell = terrain.settled.cell();
    let settled = &mut terrain.settled;
    let mut flying = Vec::new();
    for (key, part, job) in std::mem::take(&mut settled.part_jobs) {
        let made = match job.take(cx.records) {
            Taken::Waiting(job) => {
                flying.push((key, part, job));
                continue;
            }
            Taken::Ready(made) => Ok(made),
            Taken::Failed(error) if record_missing(&error) => {
                Err(Some(format!("{}: no record", PARTS.name())))
            }
            Taken::Failed(error) => Err(Some(error.to_string())),
        };
        let Some(ticket) = settled.parts.ticket(&part) else {
            continue;
        };
        let made = made.and_then(|made| {
            if made.key == key && Some(made.cell) == cell {
                Ok(made)
            } else {
                Err(None)
            }
        });
        match made {
            Ok(made) => {
                settled.parts.ready(&part, ticket, made);
            }
            Err(error) => {
                if let Some(error) = &error {
                    cx.failures.record(frame, error.clone());
                }
                let got = error.unwrap_or_else(|| "its tile or cell differs".to_owned());
                let error =
                    Error::new(ErrorCode::InvalidRecord, format!("$.jobs.{}", PARTS.name()))
                        .got(got);
                settled.parts.fail(&part, ticket, error);
            }
        }
    }
    flying.append(&mut settled.part_jobs);
    settled.part_jobs = flying;
    start_settles(terrain, cx, frame);
}

enum Start {
    Waiting,
    Started(Arc<SettleStart>),
    Broken(Option<Error>),
}

fn start_settles(terrain: &mut WaterTerrain, cx: &mut Follow<'_, '_, '_>, frame: u64) {
    let pyramid = terrain.pyramid;
    let layers = Arc::clone(&terrain.baked);
    let entity = terrain.entity;
    let mut makings = std::mem::take(&mut terrain.settled.making);
    let settled = &terrain.settled;
    let starts: Vec<Start> = makings
        .iter()
        .map(|making| {
            if making.settling {
                return Start::Waiting;
            }
            let mut parts = Vec::with_capacity(making.parts.len());
            for interest in &making.parts {
                match settled.parts.get(interest.key()) {
                    Some(Request::Ready(part)) => parts.push(part),
                    Some(Request::Failed(_) | Request::Missing) => return Start::Broken(None),
                    Some(Request::Asked(_)) | None => return Start::Waiting,
                }
            }
            let warm = |i: i64, j: i64| settled.cell_at(&pyramid, i, j);
            let warmed: Option<&dyn Fn(i64, i64) -> Option<SteadyCell>> =
                making.warm.then_some(&warm);
            match SettleStart::assemble(&layers, &pyramid, &making.region, &parts, warmed) {
                Ok(start) => Start::Started(Arc::new(start)),
                Err(error) => Start::Broken(Some(error)),
            }
        })
        .collect();
    let mut kept = Vec::with_capacity(makings.len());
    for (mut making, start) in makings.drain(..).zip(starts) {
        match start {
            Start::Waiting => kept.push(making),
            Start::Started(start) => {
                cx.queue.asked.push(SettleAsk {
                    terrain: entity,
                    inputs: making
                        .settles
                        .iter()
                        .map(|settle| Arc::clone(&settle.record.bytes))
                        .collect(),
                    start,
                });
                making.settling = true;
                making.parts.clear();
                kept.push(making);
            }
            Start::Broken(error) => {
                let error = match error {
                    Some(error) => {
                        cx.failures.record(frame, error.to_string());
                        error
                    }
                    None => {
                        Error::new(ErrorCode::InvalidRecord, format!("$.jobs.{}", PARTS.name()))
                            .got("a water part failed")
                    }
                };
                let records = &mut terrain.settled.records;
                for settle in &making.settles {
                    end(records, settle.record.key, Some(&error));
                }
                terrain.settled.changed();
            }
        }
    }
    kept.append(&mut terrain.settled.making);
    terrain.settled.making = kept;
}
