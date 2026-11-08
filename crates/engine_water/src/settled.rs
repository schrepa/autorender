use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_ecs::world::World;
use engine_core::{
    CookAsks, DeviceLimits, Error, ErrorCode, FrameBudget, FrameTime, IDLE_JOBS, Interest,
    JobFailures, Jobs, RecordKey, Records, Request, Requests, VIEW_JOBS, Views, job_priority,
};
use engine_heightfield::tile::Solid;
use engine_heightfield::{Pyramid, Stamp, reaches};
use glam::Vec2;

use crate::jobs::{
    DRAWN, REACH, ReachJob, SETTLED, WaterJob, WaterReach, WaterRegion, settled_share,
    summary_share,
};
use crate::state::{Base, Drawing, SettleQueue, WaterState, WaterTerrain};
use crate::{SettleRegion, SettledKey, SteadyCell, SteadyWater, TileKey, Water, WaterPart};

mod making;
mod steadied;

use making::{cook_regions, start_making, take_parts, take_settles};
use steadied::{Steadying, Watering, water_tiles};

#[derive(Clone, PartialEq, Eq)]
struct Input {
    key: RecordKey,
    bytes: Arc<[u8]>,
}

impl Input {
    fn of(bytes: Vec<u8>) -> Self {
        Self {
            key: RecordKey::of(&SETTLED, &bytes),
            bytes: Arc::from(bytes),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Wanted {
    record: RecordKey,
    whole: bool,
}

fn both(record: RecordKey) -> [Wanted; 2] {
    [false, true].map(|whole| Wanted { record, whole })
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Made {
    basis: [u8; 32],
    alone: bool,
    pub(crate) record: RecordKey,
}

struct Record {
    made: Made,
    summary: Interest<RecordKey>,
    whole: Option<(Arc<[u8]>, Interest<RecordKey>)>,
}

impl Record {
    fn share(&self, whole: bool) -> Option<RecordKey> {
        if whole {
            self.whole.as_ref().map(|(_, share)| *share.key())
        } else {
            Some(*self.summary.key())
        }
    }
}

#[derive(Default)]
struct Tile {
    using: Option<Interest<Wanted>>,
    asking: Vec<Interest<Wanted>>,
    whole: Option<Interest<Wanted>>,
}

struct Fetch {
    tile: TileKey,
    alone: bool,
    made: bool,
    job: Interest<RecordKey>,
}

struct Settle {
    tile: TileKey,
    alone: bool,
    record: Input,
}

struct Making {
    region: SettleRegion,
    settles: Vec<Settle>,
    warm: bool,
    parts: Vec<Interest<RecordKey>>,
    settling: bool,
}

enum Found {
    Here(RecordKey),
    Coming,
    Never,
}

struct Keyed {
    recipe: [u8; 32],
    solids: Arc<[Option<Solid>]>,
    relief: u128,
    edits: Arc<[Stamp]>,
    basis: [u8; 32],
}

struct HeldReach {
    cell: f32,
    watered: Arc<[TileKey]>,
    regions: Arc<[WaterRegion]>,
    of: Arc<[u32]>,
}

enum ReachPhase {
    Asked(Interest<RecordKey>),
    Failed,
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
            ReachPhase::Asked(_) | ReachPhase::Failed => None,
        }
    }
}

#[derive(Default)]
pub(crate) struct Shares(Requests<RecordKey, ()>);

impl Shares {
    fn hold(
        &mut self,
        jobs: &mut Jobs,
        key: RecordKey,
        bytes: impl FnOnce() -> Arc<[u8]>,
    ) -> Interest<RecordKey> {
        let interest = self.0.ask(key);
        if let Some(ticket) = self.0.ticket(&key) {
            jobs.share_as(key, bytes());
            self.0.ready(&key, ticket, ());
        }
        interest
    }

    fn keep(&mut self, key: RecordKey) -> Option<Interest<RecordKey>> {
        self.0.get(&key).and_then(Request::ready)?;
        Some(self.0.ask(key))
    }

    fn settle(&mut self, jobs: &mut Jobs, now: f64) {
        for (key, _) in self.0.settle(now) {
            jobs.unshare(key);
        }
    }
}

#[derive(Default)]
pub(crate) struct Settled {
    reach: Option<Reach>,
    keyed: Option<Keyed>,
    records: Requests<Wanted, Record>,
    tiles: BTreeMap<TileKey, Tile>,
    fetches: Vec<Fetch>,
    making: Vec<Making>,
    parts: Requests<RecordKey, WaterPart>,
    part_jobs: Vec<(TileKey, Interest<RecordKey>)>,
    waters: Requests<RecordKey, Option<Arc<[u8]>>>,
    watering: BTreeMap<TileKey, Watering>,
    steadying: Vec<Steadying>,
    pub(crate) held: BTreeMap<TileKey, (Arc<SteadyWater>, Made)>,
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
    pub(crate) fn cooking(&self) -> bool {
        self.cooking
    }

    pub(crate) fn counts(&self) -> (u32, u32) {
        let settled = self
            .tiles
            .values()
            .filter(|tile| tile.using.is_some())
            .count();
        let settling = self.tiles.values().filter(|tile| self.asked(tile)).count();
        (settled as u32, settling as u32)
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

    fn current(&self, made: &Made) -> bool {
        self.keyed
            .as_ref()
            .is_some_and(|keyed| keyed.basis == made.basis)
    }

    pub(crate) fn water_failed(&mut self, key: TileKey, error: &str) {
        let Some(watering) = self.watering.get(&key) else {
            return;
        };
        let water = *watering.water.key();
        if let Some(ticket) = self.waters.ask_again(&water) {
            let error =
                Error::new(ErrorCode::InvalidRecord, format!("$.jobs.{}", DRAWN.name())).got(error);
            self.waters.fail(&water, ticket, error);
        }
    }

    fn request(&self, interest: Option<&Interest<Wanted>>) -> Option<&Request<Record>> {
        self.records.get(interest?.key())
    }

    fn using(&self, tile: &Tile) -> Option<&Record> {
        self.request(tile.using.as_ref())?.ready()
    }

    fn asked(&self, tile: &Tile) -> bool {
        [tile.asking.last(), tile.whole.as_ref()]
            .into_iter()
            .any(|interest| matches!(self.request(interest), Some(Request::Asked(_))))
    }

    fn failed(&self, tile: &Tile) -> Option<RecordKey> {
        let interest = tile.asking.last()?;
        matches!(self.records.get(interest.key()), Some(Request::Failed(_)))
            .then_some(interest.key().record)
    }

    fn whole_ended(&self, tile: &Tile) -> bool {
        matches!(
            self.request(tile.whole.as_ref()),
            Some(Request::Failed(_) | Request::Missing)
        )
    }

    fn missing_at(&self, key: TileKey, at: usize) -> bool {
        let interest = self.tiles.get(&key).and_then(|tile| tile.asking.get(at));
        matches!(self.request(interest), Some(Request::Missing))
    }

    fn missing(&self, key: TileKey) -> bool {
        self.missing_at(key, 0)
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
            ReachPhase::Failed => return 0,
            ReachPhase::Held(held) => &held.watered,
        };
        let records = watered
            .iter()
            .filter(|&&key| {
                let single = self.single(key);
                let tile = self.tiles.get(&key);
                let current = tile
                    .and_then(|tile| self.using(tile))
                    .is_some_and(|record| {
                        self.current(&record.made) && record.made.alone == single
                    });
                !current && tile.is_none_or(|tile| self.failed(tile).is_none())
            })
            .count() as u64;
        records + self.waters_waiting + u64::from(self.waters.census().asked)
    }

    pub(crate) fn cell(&self) -> Option<f32> {
        self.held_reach().map(|reach| reach.cell)
    }

    fn cell_at(&self, pyramid: &Pyramid, i: i64, j: i64) -> Option<SteadyCell> {
        let cell = self.cell()?;
        let middle = Vec2::new((i as f32 + 0.5) * cell, (j as f32 + 0.5) * cell);
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

    fn wants_alone(&self, key: TileKey) -> bool {
        self.single(key) || (!self.cooking && self.missing(key))
    }

    fn fetched(&self, key: TileKey) -> Option<bool> {
        self.keyed.as_ref()?;
        let single = self.single(key);
        if !self.missing(key) {
            self.rect_of(key).map(|_| single)
        } else if self.cooking || single || self.missing_at(key, 1) {
            None
        } else {
            Some(true)
        }
    }

    fn done(&self, key: TileKey, whole: bool) -> bool {
        let Some(tile) = self.tiles.get(&key) else {
            return false;
        };
        let Some(using) = tile.using.as_ref().map(Interest::key) else {
            return false;
        };
        let asked = tile
            .asking
            .last()
            .is_some_and(|asked| asked.key().record == using.record);
        asked
            && self.using(tile).is_some_and(|record| {
                (using.whole || !whole)
                    && self.current(&record.made)
                    && record.made.alone == self.wants_alone(key)
            })
    }

    fn found(&self, key: TileKey, whole: bool, (made, settles): (bool, bool)) -> Found {
        let tile = self.tiles.get(&key);
        let shared = tile
            .and_then(|tile| self.using(tile))
            .and_then(|record| record.share(whole));
        if let Some(shared) = shared {
            return Found::Here(shared);
        }
        let known = self.keyed.is_some() && self.rect_of(key).is_some();
        let ended = tile
            .is_some_and(|tile| self.failed(tile).is_some() || (whole && self.whole_ended(tile)));
        let asked = tile.is_some_and(|tile| self.asked(tile));
        let coming = !ended && (asked || self.fetched(key).is_some() || (made && settles && known));
        if coming { Found::Coming } else { Found::Never }
    }
}

fn end(records: &mut Requests<Wanted, Record>, record: RecordKey, error: Option<&Error>) {
    for wanted in both(record) {
        let Some(ticket) = records.ticket(&wanted) else {
            continue;
        };
        match error {
            Some(error) => records.fail(&wanted, ticket, error.clone()),
            None => records.miss(&wanted, ticket),
        };
    }
}

fn withdraw(tile: &mut Tile, record: RecordKey) {
    let at = tile
        .asking
        .iter()
        .position(|asked| asked.key().record == record);
    if let Some(at) = at {
        tile.asking.truncate(at);
    }
    if tile
        .whole
        .as_ref()
        .is_some_and(|asked| asked.key().record == record)
    {
        tile.whole = None;
    }
}

fn asked_any(records: &Requests<Wanted, Record>, record: RecordKey) -> bool {
    both(record)
        .into_iter()
        .any(|wanted| matches!(records.get(&wanted), Some(Request::Asked(_))))
}

fn claim(
    records: &mut Requests<Wanted, Record>,
    tile: &mut Tile,
    (at, record): (usize, RecordKey),
    whole: bool,
) -> bool {
    let chain = Wanted {
        record,
        whole: false,
    };
    if tile.asking.get(at).map(|asked| *asked.key()) != Some(chain) {
        tile.asking.truncate(at);
        tile.whole = None;
        if tile.asking.len() != at {
            return false;
        }
        tile.asking.push(records.ask(chain));
    }
    if whole
        && tile
            .whole
            .as_ref()
            .is_none_or(|asked| asked.key().record != record)
    {
        tile.whole = Some(records.ask(Wanted {
            record,
            whole: true,
        }));
    }
    true
}

fn record_input(terrain: &WaterTerrain, key: TileKey, alone: bool) -> Option<Input> {
    let settled = &terrain.settled;
    let basis = settled.keyed.as_ref()?.basis;
    let tiles = if alone {
        [u32::from(key.x), u32::from(key.z), 1, 1]
    } else {
        settled.rect_of(key)?
    };
    let (low, high) = SettleRegion::alone(&terrain.pyramid, key, settled.cell()?).bounds();
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
    pub(crate) budget: &'a mut FrameBudget,
    pub(crate) failures: &'a mut JobFailures,
    pub(crate) queue: &'a mut SettleQueue,
    pub(crate) shares: &'a mut Shares,
    pub(crate) settles: bool,
}

pub(crate) fn follow(
    terrain: &mut WaterTerrain,
    water: &Water,
    cx: &mut Follow<'_, '_, '_>,
    (eye, frame, now): (Vec2, u64, f64),
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
    take_arrived(terrain, cx, &kept, frame);
    if moved {
        leave(terrain, cx, (&near, &kept, &whole), frame);
    }
    let again = std::mem::take(&mut terrain.settled.ask_again);
    if again || terrain.settled.cooking {
        ask(terrain, cx, (recipe, &watered, &near, &whole), eye);
    }
    water_tiles(terrain, cx, (&watered, &near), (eye, frame));
    settle(&mut terrain.settled, now);
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

fn settle(settled: &mut Settled, now: f64) {
    if !settled.records.settle(now).is_empty() {
        let records = &settled.records;
        settled
            .fetches
            .retain(|fetch| asked_any(records, *fetch.job.key()));
    }
    let records = &settled.records;
    settled.making.retain(|making| {
        making.settling
            || making
                .settles
                .iter()
                .any(|settle| asked_any(records, settle.record.key))
    });
    let released = settled.parts.settle(now);
    if !released.is_empty() {
        settled
            .part_jobs
            .retain(|(_, job)| !released.iter().any(|(key, _)| key == job.key()));
    }
    let released = settled.waters.settle(now);
    if !released.is_empty() {
        settled
            .steadying
            .retain(|flight| !released.iter().any(|(key, _)| key == flight.job.key()));
    }
}

fn reset(terrain: &mut WaterTerrain, reach: Option<Reach>) {
    let cooking = terrain.settled.cooking;
    terrain.settled = Settled {
        reach,
        cooking,
        ..Settled::default()
    };
    for held in terrain.held.values_mut() {
        held.given = None;
        if matches!(held.base, Base::Redrawing(_, Drawing::Given(_))) {
            held.base = Base::Made;
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
            let Some(output) = cx.records.get(ticket) else {
                return;
            };
            reach.phase = ReachPhase::Failed;
            let made = output.map_err(|error| error.to_string()).and_then(|bytes| {
                engine_codec::decode::<WaterReach>(&bytes).map_err(|error| {
                    format!("{}: its output does not decode: {error}", REACH.name())
                })
            });
            match made {
                Ok(made) => {
                    reach.phase = ReachPhase::Held(HeldReach {
                        cell: reach.wanted.1,
                        watered: made.watered.into(),
                        regions: made.regions.into(),
                        of: made.of.into(),
                    });
                    terrain.touch_all();
                }
                Err(error) => {
                    cx.failures.record(frame, error);
                }
            }
        }
        ReachPhase::Failed | ReachPhase::Held(_) => {}
    }
}

fn key_settled(terrain: &mut WaterTerrain, recipe: [u8; 32], cell: f32) {
    let solids = Arc::clone(&terrain.solids);
    let edits = Arc::clone(&terrain.edits);
    let relief = terrain.relief;
    let settled = &mut terrain.settled;
    let kept = settled.keyed.as_ref().filter(|keyed| {
        keyed.recipe == recipe && keyed.relief == relief && keyed.solids[..] == solids[..]
    });
    let edited = kept.map(|keyed| keyed.edits[..] != edits[..]);
    if edited == Some(false) {
        return;
    }
    let basis = kept.map_or_else(
        || {
            let identity = ((recipe, solids.to_vec()), (relief.to_le_bytes(), cell));
            RecordKey::shared(&engine_codec::encode(&identity)).hash()
        },
        |keyed| keyed.basis,
    );
    settled.keyed = Some(Keyed {
        recipe,
        solids,
        relief,
        edits,
        basis,
    });
    let settled = &terrain.settled;
    let restart: Vec<TileKey> = settled
        .tiles
        .iter()
        .filter(|&(&key, tile)| {
            edited.is_none() || {
                let inputs = inputs(terrain, key);
                tile.asking
                    .iter()
                    .chain(&tile.whole)
                    .any(|asked| !inputs.contains(&Some(asked.key().record)))
            }
        })
        .map(|(&key, _)| key)
        .collect();
    restart_tiles(&mut terrain.settled, restart);
    terrain.settled.changed();
}

fn inputs(terrain: &WaterTerrain, key: TileKey) -> [Option<RecordKey>; 2] {
    [false, true].map(|alone| record_input(terrain, key, alone).map(|input| input.key))
}

fn restart_tiles(settled: &mut Settled, restart: Vec<TileKey>) {
    for key in restart {
        if let Some(tile) = settled.tiles.get_mut(&key) {
            tile.asking.clear();
            tile.whole = None;
        }
    }
}

fn refresh(
    terrain: &mut WaterTerrain,
    water: &Water,
    cx: &mut Follow<'_, '_, '_>,
    frame: u64,
) -> Option<[u8; 32]> {
    let asking = (!terrain.surroundings).then_some(water);
    let (Some(water), Some(recipe)) = (asking, terrain.recipe) else {
        if terrain.settled.reach.is_some() {
            reset(terrain, None);
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
        reset(terrain, Some(Reach { wanted, phase }));
    }
    take_reach(terrain, cx, frame);
    let cell = terrain.settled.cell()?;
    key_settled(terrain, recipe, cell);
    Some(recipe)
}

fn restart_failed(terrain: &mut WaterTerrain) {
    let settled = &terrain.settled;
    let restart: Vec<TileKey> = settled
        .tiles
        .iter()
        .filter(|&(&key, tile)| {
            settled
                .failed(tile)
                .is_some_and(|failed| !inputs(terrain, key).contains(&Some(failed)))
        })
        .map(|(&key, _)| key)
        .collect();
    restart_tiles(&mut terrain.settled, restart);
}

fn ask(
    terrain: &mut WaterTerrain,
    cx: &mut Follow<'_, '_, '_>,
    (recipe, watered, near, whole): ([u8; 32], &[TileKey], &[TileKey], &[TileKey]),
    eye: Vec2,
) {
    let Some(cell) = terrain.settled.cell() else {
        return;
    };
    restart_failed(terrain);
    let pyramid = terrain.pyramid;
    let size = pyramid.tile_size(0);
    let nearness = |key: TileKey| {
        let middle = pyramid.tile_origin(key) + Vec2::splat(0.5 * size);
        (middle.distance(eye) / size) as u32
    };
    let settled = &terrain.settled;
    let cooking = settled.cooking;
    let waiting = |key: TileKey| {
        settled
            .tiles
            .get(&key)
            .is_none_or(|tile| !settled.asked(tile) && settled.failed(tile).is_none())
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
    let mut alone = Vec::new();
    for (place, key, fetched) in making {
        let Some(fetched) = fetched else {
            if cx.settles && !cooking && is_near(key) {
                alone.push((place, key));
            }
            continue;
        };
        let priority = job_priority(VIEW_JOBS, place);
        fetch_record(
            terrain,
            cx,
            (key, fetched, is_whole(key), made(key)),
            priority,
        );
    }
    if cx.settles {
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
                        .is_some_and(|(_, held)| !terrain.settled.current(held) || !held.alone);
                    (SettleRegion::alone(&pyramid, key, cell), warm, place)
                })
                .collect()
        };
        for (region, warm, place) in regions {
            start_making(
                terrain,
                cx,
                (recipe, whole),
                region,
                (!cooking, warm, place),
            );
        }
    }
    for (place, key, fetched) in fetching {
        let priority = job_priority(IDLE_JOBS, place);
        fetch_record(terrain, cx, (key, fetched, false, false), priority);
    }
}

fn fetch_record(
    terrain: &mut WaterTerrain,
    cx: &mut Follow<'_, '_, '_>,
    (key, alone, whole, made): (TileKey, bool, bool, bool),
    priority: u32,
) {
    let Some(record) = record_input(terrain, key, alone) else {
        return;
    };
    let at = usize::from(alone != terrain.settled.single(key));
    let Settled {
        records,
        tiles,
        fetches,
        ..
    } = &mut terrain.settled;
    let tile = tiles.entry(key).or_default();
    if !claim(records, tile, (at, record.key), whole) {
        return;
    }
    let flying = fetches.iter().any(|fetch| *fetch.job.key() == record.key);
    if !flying && asked_any(records, record.key) {
        let job = cx.jobs.look(&SETTLED, &record.bytes, priority);
        fetches.push(Fetch {
            tile: key,
            alone,
            made,
            job,
        });
    }
}

fn take_arrived(
    terrain: &mut WaterTerrain,
    cx: &mut Follow<'_, '_, '_>,
    kept: &[TileKey],
    frame: u64,
) {
    take_parts(terrain, cx, frame);
    let mut landed = Vec::new();
    take_fetches(terrain, cx, (kept, frame), &mut landed);
    take_settles(terrain, cx, (kept, frame), &mut landed);
    for (key, water) in landed {
        adopt(terrain, key, water, kept);
    }
}

type Landed = (TileKey, Option<(RecordKey, Arc<SteadyWater>)>);

fn take_fetches(
    terrain: &mut WaterTerrain,
    cx: &mut Follow<'_, '_, '_>,
    (kept, frame): (&[TileKey], u64),
    landed: &mut Vec<Landed>,
) {
    let mut flying = Vec::new();
    let mut taken = false;
    for fetch in std::mem::take(&mut terrain.settled.fetches) {
        if taken && cx.budget.expired() {
            flying.push(fetch);
            continue;
        }
        let output = match cx.records.request(&fetch.job) {
            None | Some(Request::Asked(_)) => {
                flying.push(fetch);
                continue;
            }
            Some(Request::Ready(bytes)) => Ok(Arc::clone(bytes)),
            Some(Request::Missing) => Err(None),
            Some(Request::Failed(error)) => Err(Some(error.clone())),
        };
        taken = true;
        let Fetch {
            tile, alone, job, ..
        } = fetch;
        let record = *job.key();
        drop(job);
        landed.extend(arrive(
            terrain,
            cx,
            (tile, alone, record),
            output,
            (kept, frame),
        ));
    }
    flying.append(&mut terrain.settled.fetches);
    terrain.settled.fetches = flying;
}

fn arrive(
    terrain: &mut WaterTerrain,
    cx: &mut Follow<'_, '_, '_>,
    (key, alone, record): (TileKey, bool, RecordKey),
    output: Result<Arc<[u8]>, Option<Error>>,
    (kept, frame): (&[TileKey], u64),
) -> Option<Landed> {
    let settled = &mut terrain.settled;
    settled.changed();
    let error = match output {
        Ok(bytes) => match land(settled, cx, (key, alone, record), (bytes, kept)) {
            Ok(water) => return Some((key, water.map(|water| (record, water)))),
            Err(Some(error)) => {
                cx.failures.record(frame, error.to_string());
                error
            }
            Err(None) => Error::new(ErrorCode::InvalidValue, format!("$.{}", SETTLED.name()))
                .got([u16::from(key.level), key.x, key.z]),
        },
        Err(None) => {
            end(&mut settled.records, record, None);
            return None;
        }
        Err(Some(error)) => {
            cx.failures
                .record(frame, format!("{}: {error}", SETTLED.name()));
            error
        }
    };
    end(&mut settled.records, record, Some(&error));
    None
}

fn land(
    settled: &mut Settled,
    cx: &mut Follow<'_, '_, '_>,
    (key, alone, record): (TileKey, bool, RecordKey),
    (bytes, kept): (Arc<[u8]>, &[TileKey]),
) -> Result<Option<Arc<SteadyWater>>, Option<Error>> {
    let [thin, wide] = both(record).map(|wanted| settled.records.ticket(&wanted).is_some());
    if !thin && !wide {
        return Ok(None);
    }
    let unread = |error: Error| Some(error.within(&format!("$.{}", SETTLED.name())));
    let (head, length) = SteadyWater::head(&bytes).map_err(unread)?;
    if head.key != key || Some(head.cell) != settled.cell() {
        return Err(None);
    }
    let Some(basis) = settled.keyed.as_ref().map(|keyed| keyed.basis) else {
        return Ok(None);
    };
    let water = if wide && kept.binary_search(&key).is_ok() {
        Some(Arc::new(
            engine_codec::decode::<SteadyWater>(&bytes).map_err(unread)?,
        ))
    } else {
        None
    };
    let made = Made {
        basis,
        alone,
        record,
    };
    let summary = cx.shares.hold(cx.jobs, summary_share(record), || {
        Arc::from(&bytes[..length])
    });
    let [thin, wide] = both(record);
    if let Some(ticket) = settled.records.ticket(&thin) {
        let value = Record {
            made,
            summary: summary.clone(),
            whole: None,
        };
        settled.records.ready(&thin, ticket, value);
    }
    if let Some(ticket) = settled.records.ticket(&wide) {
        let share = cx
            .shares
            .hold(cx.jobs, settled_share(record), || Arc::clone(&bytes));
        let value = Record {
            made,
            summary,
            whole: Some((bytes, share)),
        };
        settled.records.ready(&wide, ticket, value);
    }
    Ok(water)
}

fn adopt(
    terrain: &mut WaterTerrain,
    key: TileKey,
    water: Option<(RecordKey, Arc<SteadyWater>)>,
    kept: &[TileKey],
) {
    let Settled {
        records,
        tiles,
        held,
        ..
    } = &mut terrain.settled;
    let Some(tile) = tiles.get_mut(&key) else {
        return;
    };
    let ready = |interest: &&Interest<Wanted>| {
        records
            .get(interest.key())
            .is_some_and(|request| request.ready().is_some())
    };
    let Some(best) = tile
        .whole
        .as_ref()
        .filter(ready)
        .or_else(|| tile.asking.last().filter(ready))
    else {
        return;
    };
    if tile
        .using
        .as_ref()
        .is_some_and(|using| using.key() == best.key())
    {
        return;
    }
    let wanted = *best.key();
    tile.using = Some(best.clone());
    let made = records
        .get(&wanted)
        .and_then(Request::ready)
        .map(|record| record.made);
    if let (Some(made), Some((record, water))) = (made, water)
        && wanted.whole
        && wanted.record == record
        && kept.binary_search(&key).is_ok()
    {
        held.insert(key, (water, made));
    }
    terrain.touch(key);
}

fn leave(
    terrain: &mut WaterTerrain,
    cx: &mut Follow<'_, '_, '_>,
    (near, kept, whole): (&[TileKey], &[TileKey], &[TileKey]),
    frame: u64,
) {
    let settled = &mut terrain.settled;
    let mut dropped = Vec::new();
    settled.held.retain(|key, _| {
        let keep = kept.binary_search(key).is_ok();
        if !keep {
            dropped.push(*key);
        }
        keep
    });
    let mut demoted = Vec::new();
    for (&key, tile) in &mut settled.tiles {
        if whole.binary_search(&key).is_err() && tile.whole.take().is_some() {
            demoted.push(key);
        }
    }
    let mut stopped = false;
    if !settled.cooking {
        let gone = |key: &TileKey| near.binary_search(key).is_err();
        let Settled {
            making,
            records,
            tiles,
            fetches,
            ..
        } = &mut *settled;
        making.retain(|making| {
            let leaving = making.settles.iter().all(|settle| gone(&settle.tile));
            if leaving {
                for settle in &making.settles {
                    end(records, settle.record.key, None);
                }
                stopped = true;
            }
            !leaving
        });
        fetches.retain(|fetch| {
            let leaving = fetch.made && gone(&fetch.tile);
            if leaving && let Some(tile) = tiles.get_mut(&fetch.tile) {
                withdraw(tile, *fetch.job.key());
            }
            stopped |= leaving;
            !leaving
        });
    }
    let opened: Vec<(TileKey, Made, Result<SteadyWater, Error>)> = kept
        .iter()
        .filter(|key| !settled.held.contains_key(key))
        .filter_map(|&key| {
            let record = settled.using(settled.tiles.get(&key)?)?;
            let (bytes, _) = record.whole.as_ref()?;
            Some((key, record.made, engine_codec::decode(bytes)))
        })
        .collect();
    if !dropped.is_empty() || !demoted.is_empty() || stopped {
        settled.changed();
    }
    for (key, made, opened) in opened {
        match opened {
            Ok(water) => {
                terrain.settled.held.insert(key, (Arc::new(water), made));
                dropped.push(key);
            }
            Err(error) => cx.failures.record(frame, error.to_string()),
        }
    }
    for key in demoted {
        adopt(terrain, key, None, kept);
    }
    for key in dropped {
        terrain.touch(key);
    }
}

pub(crate) fn cook(world: &World, asks: &mut CookAsks<'_>) {
    let Some(state) = world.get_resource::<WaterState>() else {
        return;
    };
    for terrain in &state.terrains {
        if !terrain.surroundings && terrain.recipe.is_some() {
            asks.hold(state.everything.ask_later(terrain.entity));
            asks.left(u64::from(!terrain.settled.cooking));
        }
    }
}

pub(crate) fn request_settle(
    water: Res<Water>,
    views: Res<Views>,
    (time, records): (Res<FrameTime>, Res<Records>),
    limits: Res<DeviceLimits>,
    mut budget: ResMut<FrameBudget>,
    mut jobs: Jobs,
    mut state: ResMut<WaterState>,
) {
    let eye = views.main().map_or(glam::Vec3::ZERO, |main| main.detail_eye);
    let state = &mut *state;
    state.everything.settle(time.real_elapsed);
    let limits_changed = limits.is_changed();
    let mut cx = Follow {
        jobs: &mut jobs,
        records: &records,
        budget: &mut budget,
        failures: &mut state.failures,
        queue: &mut state.settling,
        shares: &mut state.shares,
        settles: limits.storage_room > 0,
    };
    let now = time.real_elapsed;
    let (mut settled_records, mut settling_records, mut settles_left) = (0, 0, 0_u64);
    for terrain in &mut state.terrains {
        if limits_changed {
            terrain.settled.stir(true);
        }
        let watering = !terrain.surroundings;
        let cooks = state.everything.interests(&terrain.entity) > 0
            && watering
            && terrain.recipe.is_some();
        if terrain.settled.cooking != cooks {
            terrain.settled.cooking = cooks;
            terrain.settled.stir(true);
        }
        if cooks {
            settles_left += terrain.settled.left();
        }
        let local = Vec2::new(eye.x - terrain.origin.x, eye.z - terrain.origin.z);
        follow(terrain, &water, &mut cx, (local, time.frame, now));
        let (settled, settling) = terrain.settled.counts();
        settled_records += settled;
        settling_records += settling;
    }
    cx.queue.settled.clear();
    cx.shares.settle(cx.jobs, now);
    state.settled_records = settled_records;
    state.settling_records = settling_records;
    state.settles_left = settles_left;
}
