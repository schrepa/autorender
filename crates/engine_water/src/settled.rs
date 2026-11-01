use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{
    CookMode, DeviceLimits, Error, FrameTime, IDLE_JOBS, JobFailures, Jobs, RecordKey, Records,
    Ticket, VIEW_JOBS, View, WorkClock, job_priority,
};
use engine_heightfield::tile::{Solid, TILE_STRIDE, TileJob};
use engine_heightfield::{Pyramid, Stamp, TILE_QUADS, reaches};
use engine_places::ground::GroundWork;
use glam::Vec2;

use crate::jobs::{PARTS, SETTLED, STEADIED, SteadiedJob, WaterJob, share_settled, share_summary};
use crate::state::{RETRY_AFTER, SettleQueue, WaterState, WaterTerrain};
use crate::water_settle::SettleAsk;
use crate::work::Work;
use crate::{
    Layers, SettleRegion, SettleStart, SettledKey, SteadyCell, SteadyWater, TileKey, TileWater,
    Water, WaterPart,
};

const TILE_ARRAY_BYTES: u64 = (TILE_STRIDE as u64) * (TILE_STRIDE as u64) * 4;
const SETTLED_TAG: u8 = 0;
const SUMMARY_TAG: u8 = 1;

pub fn settled_alike(handed: &[u8], made: &[u8]) -> bool {
    let (Ok(handed), Ok(made)) = (
        engine_codec::decode::<SteadyWater>(handed),
        engine_codec::decode::<SteadyWater>(made),
    ) else {
        return false;
    };
    if handed.key != made.key || handed.first != made.first || handed.cells != made.cells {
        return false;
    }
    let [columns, rows] = handed.cells;
    let first = handed.first;
    let (mut cells, mut wetness, mut wet, mut apart) = (0_u32, 0_u32, 0_u32, 0_u32);
    for j in 0..i64::from(rows) {
        for i in 0..i64::from(columns) {
            let at = |water: &SteadyWater| match water.at(first[0] + i, first[1] + j) {
                Some(SteadyCell::Held { level, .. }) => Some(level),
                Some(SteadyCell::Flowing { surface, .. }) => Some(surface),
                Some(SteadyCell::Dry) | None => None,
            };
            cells += 1;
            match (at(&handed), at(&made)) {
                (Some(a), Some(b)) => {
                    wet += 1;
                    apart += u32::from((a - b).abs() > 0.05);
                }
                (None, None) => {}
                (Some(_), None) | (None, Some(_)) => wetness += 1,
            }
        }
    }
    wetness * 50 <= cells && apart * 100 <= wet.max(1)
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Input {
    pub(crate) key: RecordKey,
    pub(crate) bytes: Arc<[u8]>,
}

impl Input {
    fn of(bytes: Vec<u8>) -> Self {
        Self {
            key: RecordKey::of(&SETTLED, &bytes),
            bytes: Arc::from(bytes),
        }
    }
}

enum Phase {
    Fetching(Ticket),
    Starting,
    Settling,
}

#[derive(Clone, PartialEq, Eq)]
struct Keys {
    region: Input,
    alone: Input,
}

struct Asked {
    phase: Phase,
    record: Input,
    made: bool,
}

impl Asked {
    fn starting(&self) -> bool {
        matches!(self.phase, Phase::Starting)
    }

    fn fetching(&self) -> bool {
        matches!(self.phase, Phase::Fetching(_))
    }
}

struct Making {
    region: SettleRegion,
    records: Vec<Input>,
    warm: bool,
    reached: Vec<TileKey>,
    place: u32,
}

enum Part {
    Asked(Ticket),
    Made(Arc<WaterPart>),
}

#[derive(Clone, PartialEq, Eq)]
struct Reads {
    tile: RecordKey,
    settled: Vec<(TileKey, RecordKey)>,
}

enum Wants {
    Here(Reads),
    Coming,
    Nothing,
}

enum Found {
    Here(RecordKey),
    Coming,
    Never,
}

struct Watering {
    reads: Reads,
    record: RecordKey,
    job: Option<Ticket>,
    made: Option<Arc<Option<TileWater>>>,
}

type Generation = (Option<[u8; 32]>, Arc<[Stamp]>);
type Checked = (Generation, Arc<[Option<Solid>]>, u128);

#[derive(Default)]
pub(crate) struct Settled {
    cell: f32,
    watered: Option<(Arc<Layers>, Arc<[TileKey]>)>,
    regions: Arc<[([u32; 4], Vec<TileKey>)]>,
    checked: Option<Checked>,
    records: BTreeMap<TileKey, Keys>,
    pub(crate) held: BTreeMap<TileKey, (Arc<SteadyWater>, RecordKey)>,
    shared: BTreeMap<TileKey, RecordKey>,
    summaries: BTreeMap<TileKey, RecordKey>,
    asked: BTreeMap<TileKey, Asked>,
    failed: BTreeMap<TileKey, u64>,
    missing: HashSet<RecordKey>,
    making: Vec<Making>,
    parts: BTreeMap<TileKey, Part>,
    tiles: BTreeMap<TileKey, (TileJob, RecordKey)>,
    tiles_for: Option<Generation>,
    waters: BTreeMap<TileKey, Watering>,
    waters_failed: BTreeMap<TileKey, u64>,
    given: BTreeMap<TileKey, (engine_core::Stamp, RecordKey)>,
    sharing: BTreeMap<(RecordKey, u8), RecordKey>,
    revision: u64,
    seen: Option<(u64, u64, Generation, bool)>,
    deferred: bool,
    waters_waiting: u64,
    cooking: bool,
}

impl Settled {
    pub(crate) fn counts(&self) -> (u32, u32) {
        (self.summaries.len() as u32, self.asked.len() as u32)
    }

    pub(crate) fn left(&self) -> u64 {
        let Some((_, watered)) = &self.watered else {
            return 0;
        };
        let records = watered
            .iter()
            .filter(|&&key| {
                let region = self.records.get(&key).map(|keys| keys.region.key);
                self.summaries.get(&key).copied() != region && !self.failed.contains_key(&key)
            })
            .count() as u64;
        let running = self
            .waters
            .values()
            .filter(|watering| watering.job.is_some());
        records + self.waters_waiting + running.count() as u64
    }

    pub(crate) fn cell(&self) -> f32 {
        self.cell
    }

    fn cell_at(&self, pyramid: &Pyramid, i: i64, j: i64) -> Option<SteadyCell> {
        let middle = Vec2::new((i as f32 + 0.5) * self.cell, (j as f32 + 0.5) * self.cell);
        let key = pyramid.tile_at(0, middle)?;
        self.held.get(&key)?.0.at(i, j)
    }

    pub(crate) fn ready(&self, key: TileKey, stamp: engine_core::Stamp, held_back: bool) -> bool {
        let watered = self
            .watered
            .as_ref()
            .is_some_and(|(_, keys)| keys.binary_search(&key).is_ok());
        if !watered {
            return true;
        }
        self.held.contains_key(&key)
            && (held_back
                || self
                    .given
                    .get(&key)
                    .is_some_and(|(given, _)| *given == stamp))
    }

    pub(crate) fn cancel_all(&mut self, jobs: &mut Jobs) {
        for (_, asked) in std::mem::take(&mut self.asked) {
            if let Phase::Fetching(ticket) = asked.phase {
                jobs.cancel(ticket);
            }
        }
        for (_, part) in std::mem::take(&mut self.parts) {
            if let Part::Asked(ticket) = part {
                jobs.cancel(ticket);
            }
        }
        for (_, watering) in std::mem::take(&mut self.waters) {
            if let Some(ticket) = watering.job {
                jobs.cancel(ticket);
            }
        }
        self.making.clear();
    }

    fn share(&mut self, jobs: &mut Jobs, at: (RecordKey, u8), bytes: impl FnOnce() -> Vec<u8>) {
        if self.sharing.contains_key(&at) {
            return;
        }
        let key = jobs.share(Arc::from(bytes()));
        self.sharing.insert(at, key);
    }

    fn release(&mut self, jobs: &mut Jobs, at: (RecordKey, u8)) {
        if let Some(key) = self.sharing.remove(&at) {
            jobs.unshare(key);
        }
    }

    fn release_all(&mut self, jobs: &mut Jobs) {
        for (_, key) in std::mem::take(&mut self.sharing) {
            jobs.unshare(key);
        }
    }

    fn wanted(&self, key: TileKey) -> Option<RecordKey> {
        let keys = self.records.get(&key)?;
        if self.cooking || !self.missing.contains(&keys.region.key) {
            Some(keys.region.key)
        } else {
            Some(keys.alone.key)
        }
    }

    fn fetched(&self, key: TileKey) -> Option<Input> {
        let keys = self.records.get(&key)?;
        if !self.missing.contains(&keys.region.key) {
            Some(keys.region.clone())
        } else if self.cooking || self.missing.contains(&keys.alone.key) {
            None
        } else {
            Some(keys.alone.clone())
        }
    }

    fn done(&self, key: TileKey, whole: bool) -> bool {
        let here = if whole {
            self.shared.get(&key)
        } else {
            self.summaries.get(&key)
        };
        here.copied() == self.wanted(key)
    }

    fn missing(&self, key: TileKey) -> bool {
        self.records
            .get(&key)
            .is_some_and(|keys| self.missing.contains(&keys.region.key))
    }

    fn found(&self, key: TileKey, whole: bool, (made, settles): (bool, bool)) -> Found {
        let (here, tag) = if whole {
            (self.shared.get(&key), SETTLED_TAG)
        } else {
            (self.summaries.get(&key), SUMMARY_TAG)
        };
        if let Some(&record) = here
            && let Some(&shared) = self.sharing.get(&(record, tag))
        {
            return Found::Here(shared);
        }
        let coming = !self.failed.contains_key(&key)
            && (self.asked.contains_key(&key)
                || self.fetched(key).is_some()
                || (made && settles && self.records.contains_key(&key)));
        if coming { Found::Coming } else { Found::Never }
    }

    fn fail(&mut self, keys: &[TileKey], frame: u64) {
        for &key in keys {
            self.asked.remove(&key);
            self.failed.insert(key, frame);
        }
        self.revision += 1;
    }
}

fn water_records(
    terrain: &WaterTerrain,
    recipe: [u8; 32],
    (key, tiles): (TileKey, [u32; 4]),
    cell: f32,
) -> Keys {
    let pyramid = terrain.pyramid;
    let (low, high) = SettleRegion::alone(&pyramid, key, cell).bounds();
    let mut settled = SettledKey {
        recipe,
        key,
        stamps: terrain
            .edits
            .iter()
            .copied()
            .filter(|stamp| {
                let radius = Vec2::splat(stamp.radius + stamp.falloff);
                let center = Vec2::from(stamp.center);
                let (from, to) = (center - radius, center + radius);
                from.cmple(high).all() && to.cmpge(low).all()
            })
            .collect(),
        solids: terrain.solids.to_vec(),
        relief: terrain.relief.to_le_bytes(),
        cell,
        tiles,
    };
    let region = Input::of(settled.identity());
    settled.tiles = [u32::from(key.x), u32::from(key.z), 1, 1];
    Keys {
        region,
        alone: Input::of(settled.identity()),
    }
}

fn water_part(terrain: &WaterTerrain, recipe: [u8; 32], key: TileKey, cell: f32) -> WaterJob {
    let pyramid = terrain.pyramid;
    WaterJob {
        recipe,
        key,
        stamps: terrain
            .edits
            .iter()
            .copied()
            .filter(|stamp| reaches(stamp, &pyramid, key))
            .collect(),
        solids: terrain.solids.to_vec(),
        cell,
    }
}

fn generation(terrain: &WaterTerrain) -> Generation {
    (terrain.recipe, Arc::clone(&terrain.edits))
}

fn settled_tile(terrain: &mut WaterTerrain, key: TileKey) -> Option<(TileJob, RecordKey)> {
    let basis = generation(terrain);
    if terrain.settled.tiles_for.as_ref() != Some(&basis) {
        terrain.settled.tiles = BTreeMap::new();
        terrain.settled.tiles_for = Some(basis);
    }
    if let Some(found) = terrain.settled.tiles.get(&key) {
        return Some(found.clone());
    }
    let job = terrain.tile_job(key)?;
    let record = RecordKey::of(&crate::jobs::TILE_WATER, &engine_codec::encode(&job));
    terrain.settled.tiles.insert(key, (job.clone(), record));
    Some((job, record))
}

pub(crate) struct Follow<'a, 'w, 's> {
    pub(crate) jobs: &'a mut Jobs<'w, 's>,
    pub(crate) records: &'a Records,
    pub(crate) work: &'a Work,
    pub(crate) failures: &'a mut JobFailures,
    pub(crate) queue: &'a mut SettleQueue,
    pub(crate) settles: bool,
}

pub(crate) fn follow(
    terrain: &mut WaterTerrain,
    water: &Water,
    cx: &mut Follow<'_, '_, '_>,
    (eye, frame): (Vec2, u64),
) {
    let Some((recipe, watered)) = refresh(terrain, water, cx.jobs) else {
        return;
    };
    let pyramid = terrain.pyramid;
    let near: Vec<TileKey> =
        terrain.resident_keys(|key| key.level == 0 && watered.binary_search(key).is_ok());
    let kept = beside(&pyramid, &near, &watered);
    let whole: Vec<TileKey> = if terrain.settled.cooking {
        watered.to_vec()
    } else {
        kept.clone()
    };
    take_arrived(terrain, cx, (&kept, &whole), frame);
    let settled = &mut terrain.settled;
    let held = settled.held.len();
    settled
        .held
        .retain(|key, _| kept.binary_search(key).is_ok());
    let mut gone = Vec::new();
    settled.shared.retain(|key, record| {
        let keep = whole.binary_search(key).is_ok();
        if !keep {
            gone.push((*record, SETTLED_TAG));
        }
        keep
    });
    if held != settled.held.len() || !gone.is_empty() {
        settled.revision += 1;
    }
    for at in gone {
        settled.release(cx.jobs, at);
    }
    let cooking = settled.cooking;
    let leaving: Vec<TileKey> = settled
        .asked
        .iter()
        .filter(|(key, asked)| asked.made && !cooking && near.binary_search(key).is_err())
        .map(|(&key, _)| key)
        .collect();
    for key in leaving {
        if let Some(asked) = settled.asked.remove(&key) {
            settled.revision += 1;
            if let Phase::Fetching(ticket) = asked.phase {
                cx.jobs.cancel(ticket);
            }
        }
    }
    ask(terrain, cx, (recipe, &watered, &near, &whole), (eye, frame));
    release_parts(&mut terrain.settled, cx.jobs);
    water_tiles(terrain, cx, (&watered, &near), (eye, frame));
}

fn beside(pyramid: &Pyramid, near: &[TileKey], watered: &[TileKey]) -> Vec<TileKey> {
    let [columns, rows] = pyramid.tiles(0);
    let mut tiles = Vec::with_capacity(near.len() * 2);
    for key in near {
        for dz in -1..=1_i32 {
            for dx in -1..=1_i32 {
                let (x, z) = (i32::from(key.x) + dx, i32::from(key.z) + dz);
                if x < 0 || z < 0 || x >= columns as i32 || z >= rows as i32 {
                    continue;
                }
                let at = TileKey::new(0, x as u16, z as u16);
                if watered.binary_search(&at).is_ok() {
                    tiles.push(at);
                }
            }
        }
    }
    tiles.sort_unstable();
    tiles.dedup();
    tiles
}

fn refresh(
    terrain: &mut WaterTerrain,
    water: &Water,
    jobs: &mut Jobs,
) -> Option<([u8; 32], Arc<[TileKey]>)> {
    let asking = (water.enabled && !terrain.surroundings).then_some(water);
    let baked = Arc::clone(&terrain.baked);
    let (Some(water), Some(recipe)) = (asking, terrain.recipe) else {
        terrain.settled.cancel_all(jobs);
        terrain.settled.release_all(jobs);
        let cooking = terrain.settled.cooking;
        terrain.settled = Settled {
            cooking,
            ..Settled::default()
        };
        return None;
    };
    let current = terrain.settled.cell == water.simulation_cell
        && terrain
            .settled
            .watered
            .as_ref()
            .is_some_and(|(over, _)| Arc::ptr_eq(over, &baked));
    if !current {
        terrain.settled.cancel_all(jobs);
        terrain.settled.release_all(jobs);
        let pyramid = terrain.pyramid;
        let watered: Arc<[TileKey]> = match terrain.settled.watered.take() {
            Some((over, keys)) if Arc::ptr_eq(&over, &baked) => keys,
            _ => crate::watered_tiles(&baked, &pyramid).into(),
        };
        let cell = water.simulation_cell;
        let regions = crate::settle_regions(&baked, &pyramid, cell)
            .into_iter()
            .map(|tiles| {
                let [x, z, columns, rows] = tiles;
                let inside = |key: &&TileKey| {
                    (x..x + columns).contains(&u32::from(key.x))
                        && (z..z + rows).contains(&u32::from(key.z))
                };
                (
                    tiles,
                    watered.iter().filter(inside).copied().collect::<Vec<_>>(),
                )
            })
            .filter(|(_, keys)| !keys.is_empty())
            .collect();
        let cooking = terrain.settled.cooking;
        terrain.settled = Settled {
            cell,
            watered: Some((baked, watered)),
            regions,
            cooking,
            ..Settled::default()
        };
    }
    let watered = terrain
        .settled
        .watered
        .as_ref()
        .map(|(_, keys)| keys.clone())?;
    let checked = (
        generation(terrain),
        Arc::clone(&terrain.solids),
        terrain.relief,
    );
    if terrain.settled.checked.as_ref() != Some(&checked) {
        let cell = terrain.settled.cell;
        let mut records = BTreeMap::new();
        for (tiles, keys) in terrain.settled.regions.iter() {
            for &key in keys {
                records.insert(key, water_records(terrain, recipe, (key, *tiles), cell));
            }
        }
        let settled = &mut terrain.settled;
        if settled
            .checked
            .as_ref()
            .is_some_and(|(_, solids, _)| *solids != checked.1)
        {
            for (_, part) in std::mem::take(&mut settled.parts) {
                if let Part::Asked(ticket) = part {
                    jobs.cancel(ticket);
                }
            }
        }
        settled.records = records;
        settled.checked = Some(checked);
        settled.revision += 1;
        let records = &settled.records;
        let stale: Vec<TileKey> = settled
            .asked
            .iter()
            .filter(|(key, asked)| {
                records.get(key).is_none_or(|keys| {
                    asked.record.key != keys.region.key && asked.record.key != keys.alone.key
                })
            })
            .map(|(&key, _)| key)
            .collect();
        for key in stale {
            if let Some(Asked {
                phase: Phase::Fetching(ticket),
                ..
            }) = settled.asked.remove(&key)
            {
                jobs.cancel(ticket);
            }
        }
    }
    Some((recipe, watered))
}

fn ask(
    terrain: &mut WaterTerrain,
    cx: &mut Follow<'_, '_, '_>,
    (recipe, watered, near, whole): ([u8; 32], &[TileKey], &[TileKey], &[TileKey]),
    (eye, frame): (Vec2, u64),
) {
    let pyramid = terrain.pyramid;
    let size = pyramid.tile_size(0);
    let nearness = |key: TileKey| {
        let middle = pyramid.tile_origin(key) + Vec2::splat(0.5 * size);
        (middle.distance(eye) / size) as u32
    };
    let settled = &terrain.settled;
    let cooking = settled.cooking;
    let waiting = |key: TileKey| {
        !settled.asked.contains_key(&key)
            && settled
                .failed
                .get(&key)
                .is_none_or(|&at| !cooking && frame >= at + RETRY_AFTER)
    };
    let is_near = |key: TileKey| near.binary_search(&key).is_ok();
    let is_whole = |key: TileKey| whole.binary_search(&key).is_ok();
    let made = |key: TileKey| cooking || is_near(key);
    let mut making: Vec<(u32, TileKey)> = watered
        .iter()
        .filter(|&&key| {
            (made(key) || is_whole(key)) && !settled.done(key, is_whole(key)) && waiting(key)
        })
        .map(|&key| (nearness(key), key))
        .collect();
    making.sort_unstable();
    let mut fetching: Vec<(u32, TileKey, Input)> = watered
        .iter()
        .filter(|&&key| !made(key) && !is_whole(key) && !settled.done(key, false) && waiting(key))
        .filter_map(|&key| Some((nearness(key), key, settled.fetched(key)?)))
        .collect();
    fetching.sort_unstable_by_key(|(place, key, _)| (*place, *key));
    let making: Vec<(u32, TileKey, Option<Input>)> = making
        .into_iter()
        .map(|(place, key)| (place, key, settled.fetched(key)))
        .collect();
    let cell = settled.cell;
    let mut alone = Vec::new();
    for (place, key, record) in making {
        let Some(record) = record else {
            if cx.settles
                && !cooking
                && is_near(key)
                && terrain.settled.records.contains_key(&key)
            {
                alone.push((place, key));
            }
            continue;
        };
        let priority = job_priority(VIEW_JOBS, place);
        let asked = Asked {
            phase: Phase::Fetching(cx.jobs.fetch(&SETTLED, &record.bytes, priority)),
            record,
            made: made(key),
        };
        terrain.settled.asked.insert(key, asked);
    }
    if cx.settles {
        let regions: Vec<(SettleRegion, bool, u32)> = if cooking {
            cook_regions(terrain)
        } else {
            alone
                .into_iter()
                .map(|(place, key)| {
                    let alone = terrain
                        .settled
                        .records
                        .get(&key)
                        .map(|keys| keys.alone.key);
                    let warm = terrain
                        .settled
                        .held
                        .get(&key)
                        .is_some_and(|(_, held)| Some(*held) != alone);
                    (SettleRegion::alone(&pyramid, key, cell), warm, place)
                })
                .collect()
        };
        let alone = !cooking;
        for (region, warm, place) in regions {
            start_making(&mut terrain.settled, &pyramid, region, alone, (warm, place));
        }
        ask_parts(terrain, recipe, cx.jobs);
    }
    for (place, key, record) in fetching {
        let priority = job_priority(IDLE_JOBS, place);
        let asked = Asked {
            phase: Phase::Fetching(cx.jobs.fetch(&SETTLED, &record.bytes, priority)),
            record,
            made: false,
        };
        terrain.settled.asked.insert(key, asked);
    }
}

fn cook_regions(terrain: &WaterTerrain) -> Vec<(SettleRegion, bool, u32)> {
    let pyramid = terrain.pyramid;
    let settled = &terrain.settled;
    let cell = settled.cell;
    let mut ready = Vec::new();
    for (rectangle, tiles) in settled.regions.iter() {
        let fetched = tiles.iter().all(|&key| {
            let fetching = settled.asked.get(&key).is_some_and(Asked::fetching);
            !fetching
                && (settled.done(key, false)
                    || settled.missing(key)
                    || settled.failed.contains_key(&key)
                    || settled.asked.contains_key(&key))
        });
        let kept: Vec<TileKey> = tiles
            .iter()
            .copied()
            .filter(|&key| {
                settled.missing(key)
                    && !settled.asked.contains_key(&key)
                    && !settled.failed.contains_key(&key)
            })
            .collect();
        if fetched && !kept.is_empty() {
            ready.push((
                SettleRegion::over(&pyramid, *rectangle, kept, cell),
                false,
                0,
            ));
        }
    }
    ready
}

fn start_making(
    settled: &mut Settled,
    pyramid: &Pyramid,
    region: SettleRegion,
    alone: bool,
    (warm, place): (bool, u32),
) {
    let mut records = Vec::with_capacity(region.kept.len());
    for &key in &region.kept {
        let Some(keys) = settled.records.get(&key) else {
            return;
        };
        records.push(if alone {
            keys.alone.clone()
        } else {
            keys.region.clone()
        });
    }
    for (&key, record) in region.kept.iter().zip(&records) {
        let asked = Asked {
            phase: Phase::Starting,
            record: record.clone(),
            made: true,
        };
        settled.asked.insert(key, asked);
    }
    let reached = region.reached(pyramid);
    settled.making.push(Making {
        region,
        records,
        warm,
        reached,
        place,
    });
}

fn ask_parts(terrain: &mut WaterTerrain, recipe: [u8; 32], jobs: &mut Jobs) {
    let settled = &terrain.settled;
    let mut wanted: Vec<(u32, TileKey)> = settled
        .making
        .iter()
        .flat_map(|making| making.reached.iter().map(|&key| (making.place, key)))
        .filter(|(_, key)| !settled.parts.contains_key(key))
        .collect();
    wanted.sort_unstable();
    wanted.dedup_by_key(|(_, key)| *key);
    let cell = settled.cell;
    for (place, key) in wanted {
        if terrain.settled.parts.contains_key(&key) {
            continue;
        }
        let priority = job_priority(VIEW_JOBS, place);
        let job = water_part(terrain, recipe, key, cell);
        let ticket = jobs.start(&PARTS, job.input(), priority);
        terrain.settled.parts.insert(key, Part::Asked(ticket));
    }
}

fn release_parts(settled: &mut Settled, jobs: &mut Jobs) {
    let asked = &settled.asked;
    settled.making.retain(|making| {
        making
            .region
            .kept
            .iter()
            .zip(&making.records)
            .any(|(key, record)| {
                asked
                    .get(key)
                    .is_some_and(|asked| asked.starting() && asked.record.key == record.key)
            })
    });
    let making = &settled.making;
    let parts = std::mem::take(&mut settled.parts);
    for (key, part) in parts {
        let wanted = making
            .iter()
            .any(|making| making.reached.binary_search(&key).is_ok());
        if wanted {
            settled.parts.insert(key, part);
        } else if let Part::Asked(ticket) = part {
            jobs.cancel(ticket);
        }
    }
}

fn arrived(records: &Records, ticket: &Ticket) -> Option<Result<Arc<[u8]>, Error>> {
    records.get(ticket)
}

fn take_arrived(
    terrain: &mut WaterTerrain,
    cx: &mut Follow<'_, '_, '_>,
    (kept, whole): (&[TileKey], &[TileKey]),
    frame: u64,
) {
    take_parts(terrain, cx, frame);
    let keys: Vec<TileKey> = terrain.settled.asked.keys().copied().collect();
    for key in keys {
        let Some(asked) = terrain.settled.asked.get(&key) else {
            continue;
        };
        let output = match &asked.phase {
            Phase::Fetching(ticket) => arrived(cx.records, ticket).map(|output| {
                output
                    .map(|bytes| bytes.to_vec())
                    .map_err(|_| None::<String>)
            }),
            Phase::Settling => cx
                .queue
                .take(&asked.record.bytes)
                .map(|output| output.map_err(Some)),
            Phase::Starting => None,
        };
        let Some(output) = output else {
            continue;
        };
        let Some(asked) = terrain.settled.asked.remove(&key) else {
            continue;
        };
        let fetched = matches!(asked.phase, Phase::Fetching(_));
        if let Phase::Fetching(ticket) = asked.phase {
            cx.jobs.cancel(ticket);
        }
        let settled = &mut terrain.settled;
        settled.revision += 1;
        let bytes = match output {
            Ok(bytes) => bytes,
            Err(None) if fetched => {
                settled.missing.insert(asked.record.key);
                continue;
            }
            Err(error) => {
                settled.failed.insert(key, frame);
                let error = error.unwrap_or_default();
                cx.failures
                    .record(frame, format!("{}: {error}", SETTLED.name()));
                continue;
            }
        };
        match arrive(
            &mut terrain.settled,
            cx.jobs,
            (key, asked.record.key, bytes),
            (kept, whole),
        ) {
            Ok(true) => {}
            Ok(false) => {
                terrain.settled.failed.insert(key, frame);
            }
            Err(error) => {
                terrain.settled.failed.insert(key, frame);
                cx.failures.record(frame, error.to_string());
            }
        }
    }
}

fn arrive(
    settled: &mut Settled,
    jobs: &mut Jobs,
    (key, record, bytes): (TileKey, RecordKey, Vec<u8>),
    (kept, whole): (&[TileKey], &[TileKey]),
) -> Result<bool, Error> {
    let unread = |error: Error| error.within(&format!("$.{}", SETTLED.name()));
    let (head, length) = SteadyWater::head(&bytes).map_err(unread)?;
    if head.key != key || head.cell != settled.cell {
        return Ok(false);
    }
    let held = if kept.binary_search(&key).is_ok() {
        Some(Arc::new(
            engine_codec::decode::<SteadyWater>(&bytes).map_err(unread)?,
        ))
    } else {
        None
    };
    settled.failed.remove(&key);
    settled.missing.remove(&record);
    if let Some(before) = settled.summaries.insert(key, record)
        && before != record
    {
        settled.release(jobs, (before, SUMMARY_TAG));
    }
    settled.share(jobs, (record, SUMMARY_TAG), || share_summary(&bytes[..length]));
    if whole.binary_search(&key).is_ok() {
        if let Some(before) = settled.shared.insert(key, record)
            && before != record
        {
            settled.release(jobs, (before, SETTLED_TAG));
        }
        settled.share(jobs, (record, SETTLED_TAG), || share_settled(&bytes));
    }
    if let Some(water) = held {
        settled.held.insert(key, (water, record));
    }
    Ok(true)
}

fn take_parts(terrain: &mut WaterTerrain, cx: &mut Follow<'_, '_, '_>, frame: u64) {
    let entity = terrain.entity;
    let settled = &mut terrain.settled;
    let mut broken = Vec::new();
    let keys: Vec<TileKey> = settled.parts.keys().copied().collect();
    for key in keys {
        let Some(Part::Asked(ticket)) = settled.parts.get(&key) else {
            continue;
        };
        let Some(output) = arrived(cx.records, ticket) else {
            continue;
        };
        if let Some(Part::Asked(ticket)) = settled.parts.remove(&key) {
            cx.jobs.cancel(ticket);
        }
        let made = output.map_err(|error| error.to_string()).and_then(|bytes| {
            crate::jobs::water_output(&bytes).map_err(|error| {
                format!("{}: its output does not decode: {error}", PARTS.name())
            })
        });
        match made {
            Ok(part) if part.key == key && part.cell == settled.cell => {
                settled.parts.insert(key, Part::Made(Arc::new(part)));
            }
            Ok(_) => {
                broken.push(key);
            }
            Err(error) => {
                broken.push(key);
                cx.failures.record(frame, error);
            }
        }
    }
    let pyramid = terrain.pyramid;
    let settled = &mut terrain.settled;
    let Some((layers, _)) = settled.watered.clone() else {
        return;
    };
    for making in std::mem::take(&mut settled.making) {
        if making.reached.iter().any(|key| broken.contains(key)) {
            settled.fail(&making.region.kept, frame);
            continue;
        }
        let parts: Vec<&WaterPart> = making
            .reached
            .iter()
            .filter_map(|key| match settled.parts.get(key) {
                Some(Part::Made(part)) => Some(part.as_ref()),
                Some(Part::Asked(_)) | None => None,
            })
            .collect();
        if parts.len() < making.reached.len() {
            settled.making.push(making);
            continue;
        }
        let warm = |i: i64, j: i64| settled.cell_at(&pyramid, i, j);
        let warmed: Option<&dyn Fn(i64, i64) -> Option<SteadyCell>> = making.warm.then_some(&warm);
        let start = SettleStart::assemble(&layers, &pyramid, &making.region, &parts, warmed);
        match start {
            Ok(start) => {
                for (&key, record) in making.region.kept.iter().zip(&making.records) {
                    let starting = settled.asked.get(&key).is_some_and(|asked| {
                        asked.starting() && asked.record.key == record.key
                    });
                    if starting {
                        let asked = Asked {
                            phase: Phase::Settling,
                            record: record.clone(),
                            made: true,
                        };
                        settled.asked.insert(key, asked);
                    }
                }
                cx.queue.asked.push(SettleAsk {
                    terrain: entity,
                    inputs: making.records.iter().map(|record| record.bytes.clone()).collect(),
                    start: Arc::new(start),
                });
            }
            Err(error) => {
                settled.fail(&making.region.kept, frame);
                cx.failures.record(frame, error.to_string());
            }
        }
    }
}

fn water_tiles(
    terrain: &mut WaterTerrain,
    cx: &mut Follow<'_, '_, '_>,
    (watered, near): (&[TileKey], &[TileKey]),
    (eye, frame): (Vec2, u64),
) {
    if terrain.recipe.is_none() {
        return;
    }
    take_waters(terrain, cx, frame);
    let seen = (
        terrain.settled.revision,
        terrain.revision,
        generation(terrain),
        terrain.settled.cooking,
    );
    if terrain.settled.seen.as_ref() != Some(&seen) || terrain.settled.deferred {
        terrain.settled.seen = Some(seen);
        ask_waters(terrain, cx, (watered, near), (eye, frame));
    }
    give_waters(terrain, cx.work);
}

fn take_waters(terrain: &mut WaterTerrain, cx: &mut Follow<'_, '_, '_>, frame: u64) {
    let asked: Vec<TileKey> = terrain
        .settled
        .waters
        .iter()
        .filter(|(_, watering)| watering.job.is_some())
        .map(|(&key, _)| key)
        .collect();
    let mut taken = false;
    for key in asked {
        if taken && cx.work.expired() {
            break;
        }
        let Some(ticket) = terrain
            .settled
            .waters
            .get(&key)
            .and_then(|watering| watering.job.as_ref())
        else {
            continue;
        };
        let Some(output) = arrived(cx.records, ticket) else {
            continue;
        };
        taken = true;
        if let Some(ticket) = terrain
            .settled
            .waters
            .get_mut(&key)
            .and_then(|watering| watering.job.take())
        {
            cx.jobs.cancel(ticket);
        }
        let made = output.map_err(|error| error.to_string()).and_then(|bytes| {
            crate::jobs::steadied_output(&bytes).map_err(|error| {
                format!("{}: its output does not decode: {error}", STEADIED.name())
            })
        });
        match made {
            Ok(water) => {
                let resident = terrain.resident(&key).is_some();
                if let Some(watering) = terrain.settled.waters.get_mut(&key) {
                    watering.made = resident.then(|| Arc::new(water));
                }
            }
            Err(error) => {
                terrain.settled.waters.remove(&key);
                terrain.settled.waters_failed.insert(key, frame);
                terrain.settled.deferred = true;
                cx.failures.record(frame, error);
            }
        }
    }
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
    (pyramid, key, tile): (&Pyramid, TileKey, RecordKey),
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
    Wants::Here(Reads {
        tile,
        settled: reads,
    })
}

fn ask_waters(
    terrain: &mut WaterTerrain,
    cx: &mut Follow<'_, '_, '_>,
    (watered, near): (&[TileKey], &[TileKey]),
    (eye, frame): (Vec2, u64),
) {
    let pyramid = terrain.pyramid;
    let tiles = watering(terrain, (watered, near));
    let settled = &mut terrain.settled;
    let waters = std::mem::take(&mut settled.waters);
    for (key, watering) in waters {
        if tiles.binary_search(&key).is_ok() {
            settled.waters.insert(key, watering);
        } else if let Some(ticket) = watering.job {
            cx.jobs.cancel(ticket);
        }
    }
    settled
        .waters_failed
        .retain(|key, &mut at| frame < at + RETRY_AFTER && tiles.binary_search(key).is_ok());
    settled.deferred = !settled.waters_failed.is_empty();
    let size = pyramid.tile_size(0);
    let mut wanted: Vec<(u32, TileKey)> = tiles
        .iter()
        .filter(|key| !settled.waters_failed.contains_key(key))
        .map(|&key| {
            let span = pyramid.tile_size(u32::from(key.level));
            let middle = pyramid.tile_origin(key) + Vec2::splat(0.5 * span);
            ((middle.distance(eye) / size) as u32, key)
        })
        .collect();
    wanted.sort_unstable();
    let cell = settled.cell;
    let mut waiting = 0;
    for (place, key) in wanted {
        let Some((job, tile)) = settled_tile(terrain, key) else {
            continue;
        };
        let reads = match wants(
            &terrain.settled,
            (&pyramid, key, tile),
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
        let settled = &mut terrain.settled;
        if let Some(watering) = settled.waters.get(&key)
            && watering.reads == reads
        {
            continue;
        }
        if let Some(Watering { job: Some(ticket), .. }) = settled.waters.remove(&key) {
            cx.jobs.cancel(ticket);
        }
        let priority = job_priority(VIEW_JOBS, place);
        let input = SteadiedJob {
            tile: job,
            cell,
            settled: reads
                .settled
                .iter()
                .map(|&(at, shared)| (at, shared.hash()))
                .collect(),
        }
        .input();
        let record = RecordKey::of(&STEADIED, &input);
        let ticket = cx.jobs.start(&STEADIED, input, priority);
        settled.waters.insert(
            key,
            Watering {
                reads,
                record,
                job: Some(ticket),
                made: None,
            },
        );
    }
    terrain.settled.waters_waiting = waiting;
}

fn give_waters(terrain: &mut WaterTerrain, work: &Work) {
    let resident: Vec<TileKey> = terrain.resident_keys(|_| true);
    terrain
        .settled
        .given
        .retain(|key, _| resident.binary_search(key).is_ok());
    let settled = &terrain.settled;
    type Giving = (TileKey, Arc<Option<TileWater>>, RecordKey);
    let giving: Vec<Giving> = settled
        .waters
        .iter()
        .filter_map(|(&key, watering)| {
            let water = watering.made.clone()?;
            let held = terrain.resident(&key)?;
            let held_back = held.watered.is_some();
            let over =
                settled.tiles.get(&key).map(|(_, record)| *record) == Some(watering.reads.tile);
            let record = watering.record;
            let holds = settled
                .given
                .get(&key)
                .is_some_and(|(given, made)| *given == held.face.stamp && *made == record);
            (held.face.current && !held_back && over && !holds).then_some((key, water, record))
        })
        .collect();
    let mut gave = false;
    for (key, water, record) in giving {
        if gave && (work.expired() || !work.uploads()) {
            break;
        }
        gave = true;
        work.upload(2 * TILE_ARRAY_BYTES);
        let Some(held) = terrain.held.get_mut(&key) else {
            continue;
        };
        held.hold(Option::clone(&water));
        held.watered = None;
        let stamp = held.face.stamp;
        terrain.revision += 1;
        terrain.settled.given.insert(key, (stamp, record));
    }
}

pub(crate) fn request_settle(
    water: Res<Water>,
    view: Res<View>,
    time: Res<FrameTime>,
    records: Res<Records>,
    (clock, ground, limits, cook): (
        Res<WorkClock>,
        Res<GroundWork>,
        Res<DeviceLimits>,
        Res<CookMode>,
    ),
    mut jobs: Jobs,
    mut state: ResMut<WaterState>,
) {
    let eye = view.detail_eye();
    let work = Work::of(&clock, &ground);
    let state = &mut *state;
    let mut cx = Follow {
        jobs: &mut jobs,
        records: &records,
        work: &work,
        failures: &mut state.failures,
        queue: &mut state.settling,
        settles: limits.storage_room > 0,
    };
    let (mut settled_records, mut settling_records, mut settles_left) = (0, 0, 0_u64);
    for terrain in &mut state.terrains {
        let watering = water.enabled && !terrain.surroundings;
        if cook.cooking && watering && terrain.recipe.is_some() {
            terrain.settled.cooking = true;
            settles_left += terrain.settled.left();
        }
        let local = Vec2::new(eye.x - terrain.origin.x, eye.z - terrain.origin.z);
        follow(terrain, &water, &mut cx, (local, time.frame));
        let (settled, settling) = terrain.settled.counts();
        settled_records += settled;
        settling_records += settling;
    }
    cx.queue.settled.clear();
    state.settled_records = settled_records;
    state.settling_records = settling_records;
    state.settles_left = settles_left;
}
