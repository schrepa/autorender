use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{
    CookMode, DeviceLimits, Error, FrameTime, IDLE_JOBS, JobFailures, Jobs, RecordKey, Records,
    Ticket, VIEW_JOBS, View, WorkClock, job_priority,
};
use engine_heightfield::tile::{Solid, TILE_STRIDE};
use engine_heightfield::{Pyramid, TILE_QUADS, reaches};
use engine_places::ground::GroundWork;
use glam::Vec2;

use crate::drawn::{DrawAsk, is_dry};
use crate::jobs::{
    PARTS, REACH, ReachJob, SETTLED, STEADIED, SteadiedJob, WaterJob, WaterReach, WaterRegion,
    settled_share, summary_share,
};
use crate::state::{Drawing, RETRY_AFTER, SettleQueue, WaterState, WaterTerrain};
use crate::water_settle::SettleAsk;
use crate::work::Work;
use crate::{
    Layers, SettleRegion, SettleStart, SettledKey, SteadyCell, SteadyWater, TileKey, Water,
    WaterPart,
};

const TILE_ARRAY_BYTES: u64 = (TILE_STRIDE as u64) * (TILE_STRIDE as u64) * 4;
const SETTLED_TAG: u8 = 0;
const SUMMARY_TAG: u8 = 1;

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

struct Asked {
    phase: Phase,
    record: Input,
    made: bool,
    alone: bool,
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

enum Wants {
    Here(Vec<(TileKey, RecordKey)>),
    Coming,
    Nothing,
}

enum Found {
    Here(RecordKey),
    Coming,
    Never,
}

struct Watering {
    reads: Vec<(TileKey, RecordKey)>,
    record: RecordKey,
    job: Option<Ticket>,
    made: Option<Arc<[u8]>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Made {
    generation: u64,
    alone: bool,
    pub(crate) record: RecordKey,
}

struct Keyed {
    recipe: [u8; 32],
    solids: Arc<[Option<Solid>]>,
    relief: u128,
    basis: [u8; 32],
}

struct HeldReach {
    watered: Arc<[TileKey]>,
    regions: Arc<[WaterRegion]>,
    of: Arc<[u32]>,
}

enum ReachPhase {
    Asked(Ticket),
    Failed(u64),
    Held(HeldReach),
}

struct Reach {
    wanted: ([u8; 32], f32),
    phase: ReachPhase,
}

fn reach_input((recipe, cell): ([u8; 32], f32)) -> Vec<u8> {
    engine_codec::encode(&ReachJob { recipe, cell })
}

impl Reach {
    fn held(&self) -> Option<&HeldReach> {
        match &self.phase {
            ReachPhase::Held(held) => Some(held),
            ReachPhase::Asked(_) | ReachPhase::Failed(_) => None,
        }
    }
}

#[derive(Default)]
pub(crate) struct Settled {
    cell: f32,
    reach: Option<Reach>,
    over: Option<Arc<Layers>>,
    keyed: Option<Keyed>,
    generation: u64,
    pub(crate) held: BTreeMap<TileKey, (Arc<SteadyWater>, Made)>,
    shared: BTreeMap<TileKey, Made>,
    summaries: BTreeMap<TileKey, Made>,
    asked: BTreeMap<TileKey, Asked>,
    failed: BTreeMap<TileKey, u64>,
    missing: BTreeSet<(TileKey, bool)>,
    making: Vec<Making>,
    parts: BTreeMap<TileKey, Part>,
    waters: BTreeMap<TileKey, Watering>,
    waters_failed: BTreeMap<TileKey, u64>,
    sharing: BTreeMap<(RecordKey, u8), RecordKey>,
    waters_waiting: u64,
    cooking: bool,
    near: Arc<[TileKey]>,
    kept: Arc<[TileKey]>,
    whole: Arc<[TileKey]>,
    moved: bool,
    ask_again: bool,
    water_again: bool,
    give_again: bool,
}

impl Settled {
    pub(crate) fn counts(&self) -> (u32, u32) {
        (self.summaries.len() as u32, self.asked.len() as u32)
    }

    pub(crate) fn stir(&mut self, moved: bool) {
        self.moved |= moved;
        self.ask_again |= moved;
        self.water_again = true;
        self.give_again = true;
    }

    fn changed(&mut self) {
        self.ask_again = true;
        self.water_again = true;
    }

    pub(crate) fn water_failed(&mut self, key: TileKey, frame: u64) {
        self.waters.remove(&key);
        self.waters_failed.insert(key, frame);
        self.water_again = true;
    }

    fn held_reach(&self) -> Option<&HeldReach> {
        self.reach.as_ref().and_then(Reach::held)
    }

    fn rect_of(&self, key: TileKey) -> Option<[u32; 4]> {
        let reach = self.held_reach()?;
        let at = reach.watered.binary_search(&key).ok()?;
        let region = *reach.of.get(at)? as usize;
        Some(reach.regions.get(region)?.tiles)
    }

    fn single(&self, key: TileKey) -> bool {
        self.rect_of(key)
            .is_some_and(|[_, _, columns, rows]| columns == 1 && rows == 1)
    }

    pub(crate) fn left(&self) -> u64 {
        let Some(reach) = &self.reach else {
            return 0;
        };
        let watered = match &reach.phase {
            ReachPhase::Asked(_) => return 1,
            ReachPhase::Failed(_) => return 0,
            ReachPhase::Held(held) => &held.watered,
        };
        let records = watered
            .iter()
            .filter(|&&key| {
                let single = self.single(key);
                let current = self
                    .summaries
                    .get(&key)
                    .is_some_and(|made| made.generation == self.generation && made.alone == single);
                !current && !self.failed.contains_key(&key)
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

    pub(crate) fn ready(&self, key: TileKey, given: bool) -> bool {
        let watered = self
            .held_reach()
            .is_some_and(|reach| reach.watered.binary_search(&key).is_ok());
        if !watered {
            return true;
        }
        self.held.contains_key(&key) && given
    }

    pub(crate) fn cancel_all(&mut self, jobs: &mut Jobs) {
        if let Some(Reach {
            phase: ReachPhase::Asked(ticket),
            ..
        }) = self.reach.take()
        {
            jobs.cancel(ticket);
        }
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

    fn share(
        &mut self,
        jobs: &mut Jobs,
        at: (RecordKey, u8),
        shared: impl FnOnce() -> (RecordKey, Arc<[u8]>),
    ) {
        if self.sharing.contains_key(&at) {
            return;
        }
        let (key, bytes) = shared();
        jobs.share_as(key, bytes);
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

    fn wants_alone(&self, key: TileKey) -> bool {
        self.single(key) || (!self.cooking && self.missing(key))
    }

    fn fetched(&self, key: TileKey) -> Option<bool> {
        self.keyed.as_ref()?;
        let single = self.single(key);
        if !self.missing(key) {
            self.rect_of(key).map(|_| single)
        } else if self.cooking || single || self.missing.contains(&(key, true)) {
            None
        } else {
            Some(true)
        }
    }

    fn done(&self, key: TileKey, whole: bool) -> bool {
        let here = if whole {
            self.shared.get(&key)
        } else {
            self.summaries.get(&key)
        };
        here.is_some_and(|made| {
            made.generation == self.generation && made.alone == self.wants_alone(key)
        })
    }

    fn missing(&self, key: TileKey) -> bool {
        self.missing.contains(&(key, false))
    }

    fn found(&self, key: TileKey, whole: bool, (made, settles): (bool, bool)) -> Found {
        let (here, tag) = if whole {
            (self.shared.get(&key), SETTLED_TAG)
        } else {
            (self.summaries.get(&key), SUMMARY_TAG)
        };
        if let Some(here) = here
            && let Some(&shared) = self.sharing.get(&(here.record, tag))
        {
            return Found::Here(shared);
        }
        let known = self.keyed.is_some() && self.rect_of(key).is_some();
        let coming = !self.failed.contains_key(&key)
            && (self.asked.contains_key(&key)
                || self.fetched(key).is_some()
                || (made && settles && known));
        if coming { Found::Coming } else { Found::Never }
    }

    fn fail(&mut self, keys: &[TileKey], frame: u64) {
        for &key in keys {
            self.asked.remove(&key);
            self.failed.insert(key, frame);
        }
        self.changed();
    }
}

fn record_input(terrain: &WaterTerrain, key: TileKey, alone: bool) -> Option<Input> {
    let settled = &terrain.settled;
    let basis = settled.keyed.as_ref()?.basis;
    let tiles = if alone {
        [u32::from(key.x), u32::from(key.z), 1, 1]
    } else {
        settled.rect_of(key)?
    };
    let (low, high) = SettleRegion::alone(&terrain.pyramid, key, settled.cell).bounds();
    let stamps = terrain
        .edits
        .iter()
        .copied()
        .filter(|stamp| {
            let radius = Vec2::splat(stamp.radius + stamp.falloff);
            let center = Vec2::from(stamp.center);
            let (from, to) = (center - radius, center + radius);
            from.cmple(high).all() && to.cmpge(low).all()
        })
        .collect();
    let identity = SettledKey {
        basis,
        key,
        stamps,
        tiles,
    }
    .identity();
    Some(Input::of(identity))
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

pub(crate) struct Follow<'a, 'w, 's> {
    pub(crate) jobs: &'a mut Jobs<'w, 's>,
    pub(crate) records: &'a Records,
    pub(crate) work: &'a Work,
    pub(crate) failures: &'a mut JobFailures,
    pub(crate) queue: &'a mut SettleQueue,
    pub(crate) draws: &'a mut Vec<DrawAsk>,
    pub(crate) settles: bool,
}

pub(crate) fn follow(
    terrain: &mut WaterTerrain,
    water: &Water,
    cx: &mut Follow<'_, '_, '_>,
    (eye, frame): (Vec2, u64),
) {
    let Some(recipe) = refresh(terrain, water, cx, frame) else {
        return;
    };
    let Some(watered) = terrain
        .settled
        .held_reach()
        .map(|reach| Arc::clone(&reach.watered))
    else {
        return;
    };
    let moved = std::mem::take(&mut terrain.settled.moved);
    if moved {
        let near: Vec<TileKey> =
            terrain.resident_keys(|key| key.level == 0 && watered.binary_search(key).is_ok());
        let kept: Arc<[TileKey]> = beside(&terrain.pyramid, &near, &watered).into();
        terrain.settled.whole = if terrain.settled.cooking {
            Arc::clone(&watered)
        } else {
            Arc::clone(&kept)
        };
        terrain.settled.near = near.into();
        terrain.settled.kept = kept;
    }
    let near = Arc::clone(&terrain.settled.near);
    let kept = Arc::clone(&terrain.settled.kept);
    let whole = Arc::clone(&terrain.settled.whole);
    take_arrived(terrain, cx, (&kept, &whole), frame);
    if moved {
        let settled = &mut terrain.settled;
        let mut dropped = Vec::new();
        settled.held.retain(|key, _| {
            let keep = kept.binary_search(key).is_ok();
            if !keep {
                dropped.push(*key);
            }
            keep
        });
        let mut gone = Vec::new();
        settled.shared.retain(|key, made| {
            let keep = whole.binary_search(key).is_ok();
            if !keep {
                gone.push((made.record, SETTLED_TAG));
            }
            keep
        });
        if !dropped.is_empty() || !gone.is_empty() {
            settled.changed();
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
                settled.changed();
                if let Phase::Fetching(ticket) = asked.phase {
                    cx.jobs.cancel(ticket);
                }
            }
        }
        for key in dropped {
            terrain.touch(key);
        }
    }
    let again = std::mem::take(&mut terrain.settled.ask_again);
    if again || terrain.settled.cooking || !terrain.settled.failed.is_empty() {
        ask(terrain, cx, (recipe, &watered, &near, &whole), (eye, frame));
    }
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

fn reset(terrain: &mut WaterTerrain, jobs: &mut Jobs, reach: Option<Reach>) {
    let settled = &mut terrain.settled;
    settled.cancel_all(jobs);
    settled.release_all(jobs);
    let cell = reach
        .as_ref()
        .filter(|reach| reach.held().is_some())
        .map_or(0.0, |reach| reach.wanted.1);
    *settled = Settled {
        cell,
        reach,
        cooking: settled.cooking,
        ..Settled::default()
    };
    for held in terrain.held.values_mut().chain(terrain.left.values_mut()) {
        held.given = None;
        if matches!(held.drawing, Some((_, Drawing::Given(_)))) {
            held.drawing = None;
        }
    }
    terrain.touch_all();
}

fn take_reach(terrain: &mut WaterTerrain, cx: &mut Follow<'_, '_, '_>, frame: u64) {
    let Some(reach) = terrain.settled.reach.as_mut() else {
        return;
    };
    match &reach.phase {
        ReachPhase::Asked(ticket) => {
            let Some(output) = arrived(cx.records, ticket) else {
                return;
            };
            let ReachPhase::Asked(ticket) =
                std::mem::replace(&mut reach.phase, ReachPhase::Failed(frame))
            else {
                return;
            };
            cx.jobs.cancel(ticket);
            let made = output.map_err(|error| error.to_string()).and_then(|bytes| {
                engine_codec::decode::<WaterReach>(&bytes).map_err(|error| {
                    format!("{}: its output does not decode: {error}", REACH.name())
                })
            });
            match made {
                Ok(made) => {
                    reach.phase = ReachPhase::Held(HeldReach {
                        watered: made.watered.into(),
                        regions: made.regions.into(),
                        of: made.of.into(),
                    });
                    terrain.settled.cell = reach.wanted.1;
                    terrain.touch_all();
                }
                Err(error) => {
                    cx.failures.record(frame, error);
                }
            }
        }
        ReachPhase::Failed(at) if frame >= at + RETRY_AFTER => {
            let input = reach_input(reach.wanted);
            let ticket = cx.jobs.fetch(&REACH, &input, job_priority(VIEW_JOBS, 0));
            reach.phase = ReachPhase::Asked(ticket);
        }
        ReachPhase::Failed(_) | ReachPhase::Held(_) => {}
    }
}

fn key_settled(terrain: &mut WaterTerrain, recipe: [u8; 32], jobs: &mut Jobs) {
    let solids = Arc::clone(&terrain.solids);
    let relief = terrain.relief;
    let settled = &mut terrain.settled;
    let same_solids = settled
        .keyed
        .as_ref()
        .map(|keyed| Arc::ptr_eq(&keyed.solids, &solids) || *keyed.solids == *solids);
    if let Some(keyed) = settled.keyed.as_mut()
        && same_solids == Some(true)
        && keyed.recipe == recipe
        && keyed.relief == relief
    {
        keyed.solids = solids;
        return;
    }
    if same_solids == Some(false) {
        for (_, part) in std::mem::take(&mut settled.parts) {
            if let Part::Asked(ticket) = part {
                jobs.cancel(ticket);
            }
        }
    }
    let identity = (
        (recipe, solids.to_vec()),
        (relief.to_le_bytes(), settled.cell),
    );
    let basis = RecordKey::shared(&engine_codec::encode(&identity)).hash();
    settled.keyed = Some(Keyed {
        recipe,
        solids,
        relief,
        basis,
    });
    settled.generation += 1;
    settled.missing.clear();
    for (_, asked) in std::mem::take(&mut settled.asked) {
        if let Phase::Fetching(ticket) = asked.phase {
            jobs.cancel(ticket);
        }
    }
    settled.changed();
}

fn refresh(
    terrain: &mut WaterTerrain,
    water: &Water,
    cx: &mut Follow<'_, '_, '_>,
    frame: u64,
) -> Option<[u8; 32]> {
    let asking = (water.enabled && !terrain.surroundings).then_some(water);
    let (Some(water), Some(recipe)) = (asking, terrain.recipe) else {
        if terrain.settled.reach.is_some() {
            reset(terrain, cx.jobs, None);
        }
        return None;
    };
    let wanted = (recipe, water.simulation_cell);
    if terrain
        .settled
        .reach
        .as_ref()
        .is_none_or(|reach| reach.wanted != wanted)
    {
        let input = reach_input(wanted);
        let ticket = cx.jobs.fetch(&REACH, &input, job_priority(VIEW_JOBS, 0));
        let phase = ReachPhase::Asked(ticket);
        reset(terrain, cx.jobs, Some(Reach { wanted, phase }));
    }
    take_reach(terrain, cx, frame);
    terrain.settled.held_reach()?;
    let baked = Arc::clone(&terrain.baked);
    if terrain
        .settled
        .over
        .as_ref()
        .is_some_and(|over| !Arc::ptr_eq(over, &baked))
    {
        let reach = terrain.settled.reach.take();
        reset(terrain, cx.jobs, reach);
    }
    terrain.settled.over = Some(baked);
    key_settled(terrain, recipe, cx.jobs);
    Some(recipe)
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
    let mut fetching: Vec<(u32, TileKey, bool)> = watered
        .iter()
        .filter(|&&key| !made(key) && !is_whole(key) && !settled.done(key, false) && waiting(key))
        .filter_map(|&key| Some((nearness(key), key, settled.fetched(key)?)))
        .collect();
    fetching.sort_unstable();
    let making: Vec<(u32, TileKey, Option<bool>)> = making
        .into_iter()
        .map(|(place, key)| (place, key, settled.fetched(key)))
        .collect();
    let cell = settled.cell;
    let mut alone = Vec::new();
    for (place, key, fetched) in making {
        let Some(fetched) = fetched else {
            if cx.settles && !cooking && is_near(key) {
                alone.push((place, key));
            }
            continue;
        };
        let priority = job_priority(VIEW_JOBS, place);
        fetch_record(terrain, cx, (key, fetched, made(key)), priority);
    }
    if cx.settles {
        let generation = terrain.settled.generation;
        let regions: Vec<(SettleRegion, bool, u32)> = if cooking {
            cook_regions(terrain)
        } else {
            alone
                .into_iter()
                .map(|(place, key)| {
                    let warm = terrain
                        .settled
                        .held
                        .get(&key)
                        .is_some_and(|(_, held)| held.generation != generation || !held.alone);
                    (SettleRegion::alone(&pyramid, key, cell), warm, place)
                })
                .collect()
        };
        let alone = !cooking;
        for (region, warm, place) in regions {
            start_making(terrain, region, alone, (warm, place));
        }
        ask_parts(terrain, recipe, cx.jobs);
    }
    for (place, key, fetched) in fetching {
        let priority = job_priority(IDLE_JOBS, place);
        fetch_record(terrain, cx, (key, fetched, false), priority);
    }
}

fn fetch_record(
    terrain: &mut WaterTerrain,
    cx: &mut Follow<'_, '_, '_>,
    (key, alone, made): (TileKey, bool, bool),
    priority: u32,
) {
    let Some(record) = record_input(terrain, key, alone) else {
        return;
    };
    let asked = Asked {
        phase: Phase::Fetching(cx.jobs.fetch(&SETTLED, &record.bytes, priority)),
        record,
        made,
        alone,
    };
    terrain.settled.asked.insert(key, asked);
}

fn cook_regions(terrain: &WaterTerrain) -> Vec<(SettleRegion, bool, u32)> {
    let pyramid = terrain.pyramid;
    let settled = &terrain.settled;
    let cell = settled.cell;
    let Some(reach) = settled.held_reach() else {
        return Vec::new();
    };
    let mut ready = Vec::new();
    for region in reach.regions.iter() {
        let (rectangle, tiles) = (region.tiles, &region.keys);
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
                SettleRegion::over(&pyramid, rectangle, kept, cell),
                false,
                0,
            ));
        }
    }
    ready
}

fn start_making(
    terrain: &mut WaterTerrain,
    region: SettleRegion,
    alone: bool,
    (warm, place): (bool, u32),
) {
    let mut asking = Vec::with_capacity(region.kept.len());
    for &key in &region.kept {
        let alone = alone || terrain.settled.single(key);
        let Some(record) = record_input(terrain, key, alone) else {
            return;
        };
        asking.push((record, alone));
    }
    let settled = &mut terrain.settled;
    for (&key, (record, alone)) in region.kept.iter().zip(&asking) {
        let asked = Asked {
            phase: Phase::Starting,
            record: record.clone(),
            made: true,
            alone: *alone,
        };
        settled.asked.insert(key, asked);
    }
    let reached = region.reached(&terrain.pyramid);
    settled.making.push(Making {
        region,
        records: asking.into_iter().map(|(record, _)| record).collect(),
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
    let mut taken = false;
    for key in keys {
        let Some(asked) = terrain.settled.asked.get(&key) else {
            continue;
        };
        let output = match &asked.phase {
            Phase::Fetching(_) if taken && cx.work.expired() => continue,
            Phase::Fetching(ticket) => {
                arrived(cx.records, ticket).map(|output| output.map_err(|_| None::<String>))
            }
            Phase::Settling => cx
                .queue
                .take(&asked.record.bytes)
                .map(|output| output.map(Arc::from).map_err(Some)),
            Phase::Starting => None,
        };
        let Some(output) = output else {
            continue;
        };
        let Some(asked) = terrain.settled.asked.remove(&key) else {
            continue;
        };
        taken = true;
        let fetched = matches!(asked.phase, Phase::Fetching(_));
        if let Phase::Fetching(ticket) = asked.phase {
            cx.jobs.cancel(ticket);
        }
        let settled = &mut terrain.settled;
        settled.changed();
        let missed = (key, asked.alone && !settled.single(key));
        let bytes = match output {
            Ok(bytes) => bytes,
            Err(None) if fetched => {
                settled.missing.insert(missed);
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
        let made = Made {
            generation: settled.generation,
            alone: asked.alone,
            record: asked.record.key,
        };
        match arrive(settled, cx.jobs, (key, made, bytes), (kept, whole)) {
            Ok(true) => {
                terrain.settled.missing.remove(&missed);
                terrain.touch(key);
            }
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
    (key, made, bytes): (TileKey, Made, Arc<[u8]>),
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
    let record = made.record;
    if let Some(before) = settled.summaries.insert(key, made)
        && before.record != record
    {
        settled.release(jobs, (before.record, SUMMARY_TAG));
    }
    settled.share(jobs, (record, SUMMARY_TAG), || {
        (summary_share(record), Arc::from(&bytes[..length]))
    });
    if whole.binary_search(&key).is_ok() {
        if let Some(before) = settled.shared.insert(key, made)
            && before.record != record
        {
            settled.release(jobs, (before.record, SETTLED_TAG));
        }
        settled.share(jobs, (record, SETTLED_TAG), || {
            (settled_share(record), Arc::clone(&bytes))
        });
    }
    if let Some(water) = held {
        settled.held.insert(key, (water, made));
    }
    Ok(true)
}

fn take_parts(terrain: &mut WaterTerrain, cx: &mut Follow<'_, '_, '_>, frame: u64) {
    let entity = terrain.entity;
    let layers = Arc::clone(&terrain.baked);
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
            crate::jobs::water_output(&bytes)
                .map_err(|error| format!("{}: its output does not decode: {error}", PARTS.name()))
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
                    if let Some(asked) = settled.asked.get_mut(&key)
                        && asked.starting()
                        && asked.record.key == record.key
                    {
                        asked.phase = Phase::Settling;
                    }
                }
                cx.queue.asked.push(SettleAsk {
                    terrain: entity,
                    inputs: making
                        .records
                        .iter()
                        .map(|record| record.bytes.clone())
                        .collect(),
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
    let took = take_waters(terrain, cx, frame);
    let asks = std::mem::take(&mut terrain.settled.water_again);
    if asks {
        ask_waters(terrain, cx, (watered, near), (eye, frame));
        terrain.settled.water_again = !terrain.settled.waters_failed.is_empty();
    }
    let gives = std::mem::take(&mut terrain.settled.give_again);
    if took || asks || gives {
        terrain.settled.give_again = give_waters(terrain, cx.work, cx.draws);
    }
}

fn take_waters(terrain: &mut WaterTerrain, cx: &mut Follow<'_, '_, '_>, frame: u64) -> bool {
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
            is_dry(&bytes).map(|_| bytes).map_err(|error| {
                format!("{}: its output does not decode: {error}", STEADIED.name())
            })
        });
        match made {
            Ok(bytes) => {
                let resident = terrain.resident(&key).is_some();
                if let Some(watering) = terrain.settled.waters.get_mut(&key) {
                    watering.made = resident.then_some(bytes);
                }
            }
            Err(error) => {
                terrain.settled.water_failed(key, frame);
                cx.failures.record(frame, error);
            }
        }
    }
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
        if terrain
            .settled
            .waters
            .get(&key)
            .is_some_and(|watering| watering.reads == reads)
        {
            continue;
        }
        let Some(tile) = terrain.tile_job(key) else {
            continue;
        };
        let settled = &mut terrain.settled;
        if let Some(Watering {
            job: Some(ticket), ..
        }) = settled.waters.remove(&key)
        {
            cx.jobs.cancel(ticket);
        }
        let priority = job_priority(VIEW_JOBS, place);
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

fn give_waters(terrain: &mut WaterTerrain, work: &Work, draws: &mut Vec<DrawAsk>) -> bool {
    let giving: Vec<(TileKey, Arc<[u8]>, RecordKey)> = terrain
        .settled
        .waters
        .iter()
        .filter_map(|(&key, watering)| {
            let bytes = watering.made.clone()?;
            let held = terrain.resident(&key)?;
            let record = watering.record;
            let coming =
                matches!(&held.drawing, Some((_, Drawing::Given(given))) if *given == record);
            let holds = coming || held.given == Some(record);
            (held.face.current && held.watered.is_none() && !holds).then_some((key, bytes, record))
        })
        .collect();
    let entity = terrain.entity;
    let mut gave = false;
    for (key, bytes, record) in giving {
        if gave && (work.expired() || !work.uploads()) {
            return true;
        }
        gave = true;
        work.upload(2 * TILE_ARRAY_BYTES);
        let Some(held) = terrain.held.get_mut(&key) else {
            continue;
        };
        if matches!(is_dry(&bytes), Ok(true)) {
            held.water = None;
            held.drawn = None;
            held.watered = None;
            held.given = Some(record);
            held.drawing = None;
            terrain.touch(key);
        } else {
            draws.push(DrawAsk {
                terrain: entity,
                key,
                stamp: held.face.stamp,
                record: bytes,
                drawing: Drawing::Given(record),
            });
        }
    }
    false
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
    let state = &mut *state;
    let work = Work::until(&clock, &ground, state.work_end);
    let limits_changed = limits.is_changed();
    let mut cx = Follow {
        jobs: &mut jobs,
        records: &records,
        work: &work,
        failures: &mut state.failures,
        queue: &mut state.settling,
        draws: &mut state.draw_asks,
        settles: limits.storage_room > 0,
    };
    let (mut settled_records, mut settling_records, mut settles_left) = (0, 0, 0_u64);
    for terrain in &mut state.terrains {
        if limits_changed {
            terrain.settled.stir(true);
        }
        let watering = water.enabled && !terrain.surroundings;
        if cook.cooking && watering && terrain.recipe.is_some() {
            if !terrain.settled.cooking {
                terrain.settled.cooking = true;
                terrain.settled.stir(true);
            }
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
