//! Terrains' settled water: each level-0 tile's rivers at their steady
//! state ([`SteadyWater`]), records the cook makes for every tile its
//! bake's water reaches and hosts keep, at the cells' size water's
//! simulation takes ([`Water::simulation_cell`]). Frames ask for the
//! records of the resident level-0 tiles water reaches, fetch every other
//! tile's where a host holds it, nearest first; they keep resident level-0
//! tiles' records whole and every record's summary ([`SteadySummary`]).
//! Resident level-0 tiles take the water their records and their
//! neighbors' stand at their samples ([`Tile::steadied`]), coarser tiles
//! the flow and foam of the summaries under them ([`Tile::flowing`]):
//! drawing, queries and the window simulated around the camera read water
//! as it settles, near and far. A record made under other edits than the
//! ground has now stands until the one made under them arrives. A tile
//! whose rivers something standing in them holds back keeps the water they
//! run with now.
//!
//! A record no host holds is made in two parts: a job prepares what its
//! water settles from ([`WaterStart`], on a worker), from the record it
//! replaces where one stands, and the renderer settles it on the GPU,
//! handing the record to its host to keep ([`SettleQueue`]); records the
//! GPU made are cooked once, and a verifying host compares those made
//! again within their tolerance ([`settled_alike`]). While a session
//! cooks, every watered tile's record is made, nearest first.

use super::*;
use crate::{MISSING_RECORD, RecordParity};
use engine_terrain::jobs::WaterJob;
use engine_terrain::{
    SUMMARY_SIDE, SettleStart, SteadyCell, SteadySquare, SteadySummary, SteadyWater,
};

/// The kind settled water's records are kept by.
pub const SETTLED_KIND: &str = "settled_water";

/// How a verifying host compares settled water made again on the GPU with
/// the record it handed back ([`settled_alike`]).
pub const SETTLED_PARITY: RecordParity = RecordParity::Within(settled_alike);

/// Whether settled water `made` again keeps within its tolerance of the
/// record `handed`: the same tile and cells, wet and dry alike at all but
/// one cell in fifty, and the cells wet in both standing within five
/// centimeters of each other at all but one in a hundred.
pub fn settled_alike(handed: &[u8], made: &[u8]) -> bool {
    let (Ok(handed), Ok(made)) = (
        engine_terrain::codec::decode::<SteadyWater>(handed),
        engine_terrain::codec::decode::<SteadyWater>(made),
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

/// Where a record of settled water asked for stands: fetched from a host's
/// packs and store, its start prepared by a worker (each by its job), or
/// settling on the GPU.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Fetching(JobId),
    Starting(JobId),
    Settling,
}

/// A record of settled water asked for: where it stands, its key, and
/// whether it is made where no host holds it (or only fetched, for its
/// summary).
#[derive(Clone, Copy)]
struct Asked {
    phase: Phase,
    record: RecordKey,
    made: bool,
}

/// Water settling on the GPU, which terrains share: the hash of the
/// renderer's code settling it, which its records are keyed by with their
/// starts' ([`Terrains::set_water_settling`]), and whether a renderer
/// settles it here; the starts handed to the renderer, each with its
/// record's key, until its host takes them; and what the renderer settled,
/// by key, for the frame's terrains to take (each a record, or why it
/// failed).
#[derive(Default)]
pub(super) struct SettleQueue {
    pub(super) source: [u8; 32],
    pub(super) settles: bool,
    asked: Vec<(RecordKey, Arc<SettleStart>)>,
    settled: Vec<(RecordKey, Result<Vec<u8>, String>)>,
}

impl SettleQueue {
    /// What the renderer settled as `record`, taken.
    fn take(&mut self, record: RecordKey) -> Option<Result<Vec<u8>, String>> {
        let at = self.settled.iter().position(|(held, _)| *held == record)?;
        Some(self.settled.swap_remove(at).1)
    }

    /// Drops what the renderer settled that no terrain took this frame:
    /// records none asks for any more.
    pub(super) fn drop_untaken(&mut self) {
        self.settled.clear();
    }
}

impl Terrains {
    /// Has water settle on the GPU with code hashing to `source`, which
    /// settled water's records are keyed by, where a renderer `settles` it
    /// (without one, records are only fetched).
    pub fn set_water_settling(&mut self, source: [u8; 32], settles: bool) {
        self.settling.source = source;
        self.settling.settles = settles;
    }

    /// The starts of water to settle handed to the renderer since the last
    /// call, each with its record's key.
    pub fn take_water_settles(&mut self) -> Vec<(RecordKey, Arc<SettleStart>)> {
        std::mem::take(&mut self.settling.asked)
    }

    /// The renderer's settled water of `record`: its record's bytes, or why
    /// it failed.
    pub fn water_settled(&mut self, record: RecordKey, settled: Result<Vec<u8>, String>) {
        self.settling.settled.push((record, settled));
    }
}

/// A terrain's settled water: the cells' size its records hold; the
/// level-0 tiles its bake's water reaches, found over its baked layers;
/// the generation of tiles its records' keys were found for, and each
/// watered tile's record's key then; the records of resident level-0 tiles
/// held whole and every record's summary, each with its key; the records
/// asked for, those that failed (at the frame they did) and those no host
/// holds; per tile, the latest of the records its samples read to arrive,
/// arrivals counted; the resident tiles given their water, as given and
/// when; and whether its session cooks, which makes every watered tile's.
#[derive(Default)]
pub(super) struct Settled {
    cell: f32,
    watered: Option<(Arc<Layers>, Arc<[TileKey]>)>,
    checked: Option<Generation>,
    records: TileMap<RecordKey>,
    pub(super) held: TileMap<(Arc<SteadyWater>, RecordKey)>,
    summaries: TileMap<(SteadySummary, RecordKey)>,
    asked: TileMap<Asked>,
    failed: TileMap<u64>,
    missing: TileMap<RecordKey>,
    changed: TileMap<u64>,
    given: TileMap<(Arc<Tile>, u64)>,
    arrived: u64,
    pub(super) cooking: bool,
}

impl Settled {
    /// The jobs of its records asked for.
    pub(super) fn asked(&self) -> Vec<JobId> {
        self.asked
            .iter()
            .filter_map(|(_, asked)| match asked.phase {
                Phase::Fetching(id) | Phase::Starting(id) => Some(id),
                Phase::Settling => None,
            })
            .collect()
    }

    /// Records arrived, whose summaries it holds, and those asked for.
    pub(super) fn counts(&self) -> (u32, u32) {
        (self.summaries.len() as u32, self.asked.len() as u32)
    }

    /// Watered tiles whose record under the edits the ground has now has
    /// not arrived and did not fail: what a cook has left.
    pub(super) fn left(&self) -> u64 {
        let Some((_, watered)) = &self.watered else {
            return 0;
        };
        watered
            .iter()
            .filter(|&&key| {
                self.summaries.get(&key).map(|(_, record)| record) != self.records.get(&key)
                    && !self.failed.contains_key(&key)
            })
            .count() as u64
    }

    /// The meters across its records' cells.
    pub(super) fn cell(&self) -> f32 {
        self.cell
    }

    /// Whether the settled water of level-0 tile `key` is known: its record
    /// is held whole, or its bake's water reaches it not, so it settles
    /// dry.
    pub(super) fn has(&self, key: TileKey) -> bool {
        self.held.contains_key(&key)
            || self
                .watered
                .as_ref()
                .is_some_and(|(_, watered)| watered.binary_search(&key).is_err())
    }

    /// The settled water of the terrain's lattice cell `(i, j)`: what the
    /// record of the level-0 tile holding its middle says, if held whole.
    pub(super) fn cell_at(&self, pyramid: &Pyramid, i: i64, j: i64) -> Option<SteadyCell> {
        let middle = Vec2::new((i as f32 + 0.5) * self.cell, (j as f32 + 0.5) * self.cell);
        let key = pyramid.tile_at(0, middle)?;
        self.held.get(&key)?.0.at(i, j)
    }

    /// Square `(i, j)` of the terrain's lattice of summaries' squares,
    /// [`SUMMARY_SIDE`] along each side of a level-0 tile: what the summary
    /// of the tile holding it says, if held.
    fn square_at(&self, i: i64, j: i64) -> Option<SteadySquare> {
        let side = i64::from(SUMMARY_SIDE);
        if i < 0 || j < 0 {
            return None;
        }
        let x = u16::try_from(i / side).ok()?;
        let z = u16::try_from(j / side).ok()?;
        let (summary, _) = self.summaries.get(&TileKey::new(0, x, z))?;
        summary.square((i % side) as u32, (j % side) as u32)
    }

    /// Whether level-0 tile `key`'s record under the edits the ground has
    /// now arrived as a tile `near` keeps it: held whole, or for another
    /// tile its summary.
    fn done(&self, key: TileKey, near: bool) -> bool {
        let record = self.records.get(&key);
        if near {
            self.held.get(&key).map(|(_, held)| held) == record
        } else {
            self.summaries.get(&key).map(|(_, held)| held) == record
        }
    }

    /// Counts a record of level-0 tile `key` of `pyramid` arrived: the
    /// tiles whose samples read it change, the tile and its neighbors and
    /// every tile over them.
    fn arrive(&mut self, pyramid: &Pyramid, key: TileKey) {
        self.arrived += 1;
        let [columns, rows] = pyramid.tiles(0);
        for dz in -1..=1_i32 {
            for dx in -1..=1_i32 {
                let (x, z) = (i32::from(key.x) + dx, i32::from(key.z) + dz);
                if x < 0 || z < 0 || x >= columns as i32 || z >= rows as i32 {
                    continue;
                }
                let mut over = Some(TileKey::new(0, x as u16, z as u16));
                while let Some(tile) = over {
                    self.changed.insert(tile, self.arrived);
                    over = pyramid.parent(tile);
                }
            }
        }
    }
}

impl TerrainState {
    /// The job preparing level-0 tile `key`'s water to settle over the
    /// stack shared as `stack`, carrying the edits reaching the ground it
    /// settles over, in cells `cell` meters across, from `warm`; and the key
    /// of its record, settled by code hashing to `settling` with the
    /// start's own.
    pub(super) fn water_record(
        &self,
        stack: &str,
        key: TileKey,
        (cell, settling): (f32, &[u8; 32]),
        warm: Option<SteadyWater>,
    ) -> (WaterJob, RecordKey) {
        let pyramid = self.terrain.pyramid;
        let (low, high) = engine_terrain::steady_water_bounds(&pyramid, key, cell);
        let job = WaterJob {
            stack: stack.to_owned(),
            pyramid,
            key,
            stamps: self
                .terrain
                .edits
                .iter()
                .map(|edit| edit.stamp)
                .filter(|stamp| {
                    let radius = Vec2::splat(stamp.radius + stamp.falloff);
                    let center = Vec2::from(stamp.center);
                    let (from, to) = (center - radius, center + radius);
                    from.cmple(high).all() && to.cmpge(low).all()
                })
                .collect(),
            cell,
            warm,
        };
        let identity = [settling.as_slice(), &job.identity()].concat();
        let record = RecordKey::new(&engine_terrain::sources::WATER, SETTLED_KIND, &identity);
        (job, record)
    }
}

/// Cancels every record `state`'s settled water asked for, giving `slots`
/// theirs back.
fn cancel_all(state: &TerrainState, slots: &mut JobBudget<'_>) {
    for id in state.settled.asked() {
        slots.jobs.cancel(id);
        slots.release();
    }
}

/// Follows `state`'s settled water for this frame, where `water` asks for
/// water: takes the records that arrived (fetched, or settled on the GPU
/// through `queue`); asks for the records of the resident level-0 tiles
/// its bake's water reaches (every watered tile's while its session cooks)
/// and fetches the other watered tiles' while `slots` admits them, nearest
/// `eye` (terrain-local) first, those of tiles no longer resident made no
/// more; and gives resident tiles the water their records stand. Without
/// water, or without a bake's water, it holds nothing.
pub(super) fn follow(
    state: &mut TerrainState,
    water: Option<&Water>,
    slots: &mut JobBudget<'_>,
    queue: &mut SettleQueue,
    (eye, frame): (Vec2, u64),
) {
    let Some((stack, watered)) = refresh(state, water, slots, queue) else {
        return;
    };
    let near: Vec<TileKey> = state
        .store
        .keys_where(|key| key.level == 0 && watered.binary_search(key).is_ok());
    take_arrived(state, slots, queue, &near, frame);
    let settled = &mut state.settled;
    settled
        .held
        .retain(|key, _| near.binary_search(key).is_ok());
    let cooking = settled.cooking;
    let leaving = settled.asked.keys_where(|key, asked| {
        asked.made && !cooking && near.binary_search(key).is_err()
    });
    for key in leaving {
        if let Some(asked) = settled.asked.remove(&key) {
            match asked.phase {
                Phase::Fetching(id) | Phase::Starting(id) => {
                    slots.jobs.cancel(id);
                    slots.release();
                }
                // What the GPU settles still comes, for its host to keep.
                Phase::Settling => {}
            }
        }
    }
    ask(state, slots, queue, (&stack, &watered, &near), (eye, frame));
    give(state, &near);
}

/// Readies `state`'s settled water for `water`: none without water or a
/// bake's water (what it asked for cancelled, `slots` given theirs back),
/// anew over other baked layers or cells' size, and each watered tile's
/// record's key found again under other edits (keyed by `queue`'s
/// settling), the records asked for under others cancelled. The stack its
/// records settle over and the level-0 tiles its bake's water reaches, in
/// key order.
fn refresh(
    state: &mut TerrainState,
    water: Option<&Water>,
    slots: &mut JobBudget<'_>,
    queue: &SettleQueue,
) -> Option<(String, Arc<[TileKey]>)> {
    let asking = water.filter(|water| water.enabled && state.around.is_none());
    let (Some(water), Some(stack), Some(baked)) = (
        asking,
        state.recipe.as_ref().map(|recipe| recipe.stack_key.clone()),
        state.baked.clone(),
    ) else {
        cancel_all(state, slots);
        let cooking = state.settled.cooking;
        state.settled = Settled {
            cooking,
            ..Settled::default()
        };
        return None;
    };
    let current = state.settled.cell == water.simulation_cell
        && state
            .settled
            .watered
            .as_ref()
            .is_some_and(|(over, _)| Arc::ptr_eq(over, &baked));
    if !current {
        cancel_all(state, slots);
        let watered: Arc<[TileKey]> = match state.settled.watered.take() {
            Some((over, keys)) if Arc::ptr_eq(&over, &baked) => keys,
            _ => engine_terrain::watered_tiles(&baked, &state.terrain.pyramid).into(),
        };
        let cooking = state.settled.cooking;
        state.settled = Settled {
            cell: water.simulation_cell,
            watered: Some((baked, watered)),
            cooking,
            ..Settled::default()
        };
    }
    let watered = state
        .settled
        .watered
        .as_ref()
        .map(|(_, keys)| keys.clone())?;
    let generation = state.store.generation();
    if state.settled.checked != Some(generation) {
        let cell = state.settled.cell;
        let mut records = TileMap::default();
        for &key in watered.iter() {
            let (_, record) = state.water_record(&stack, key, (cell, &queue.source), None);
            records.insert(key, record);
        }
        let settled = &mut state.settled;
        settled.records = records;
        settled.checked = Some(generation);
        let records = &settled.records;
        let stale = settled
            .asked
            .keys_where(|key, asked| records.get(key) != Some(&asked.record));
        for key in stale {
            if let Some(asked) = settled.asked.remove(&key) {
                match asked.phase {
                    Phase::Fetching(id) | Phase::Starting(id) => {
                        slots.jobs.cancel(id);
                        slots.release();
                    }
                    Phase::Settling => {}
                }
            }
        }
    }
    Some((stack, watered))
}

/// Asks, while `slots` admits them, for the records over the stack shared
/// as `stack` that `state`'s settled water lacks under the edits the ground
/// has now, nearest `eye` (terrain-local) first: those of the resident
/// level-0 tiles `near` (every tile of `watered` while its session cooks),
/// fetched where a host holds them and else made, their water prepared to
/// settle (from the record a tile holds under other edits) where `queue`
/// has a renderer settle it; then those of the other tiles of `watered`,
/// fetched where a host holds them. A record that failed is asked for again
/// a while after it did, `frame` now.
fn ask(
    state: &mut TerrainState,
    slots: &mut JobBudget<'_>,
    queue: &SettleQueue,
    (stack, watered, near): (&str, &[TileKey], &[TileKey]),
    (eye, frame): (Vec2, u64),
) {
    let pyramid = state.terrain.pyramid;
    let size = pyramid.tile_size(0);
    let nearness = |key: TileKey| {
        let middle = pyramid.tile_origin(key) + Vec2::splat(0.5 * size);
        (middle.distance(eye) / size) as u32
    };
    let settled = &state.settled;
    let cooking = settled.cooking;
    let waiting = |key: TileKey| {
        !settled.asked.contains_key(&key)
            && settled
                .failed
                .get(&key)
                .is_none_or(|&at| !cooking && frame >= at + RETRY_AFTER)
    };
    let is_near = |key: TileKey| near.binary_search(&key).is_ok();
    let made = |key: TileKey| cooking || is_near(key);
    let mut making: Vec<(u32, TileKey)> = watered
        .iter()
        .filter(|&&key| made(key) && !settled.done(key, is_near(key)) && waiting(key))
        .map(|&key| (nearness(key), key))
        .collect();
    making.sort_unstable();
    let mut fetching: Vec<(u32, TileKey)> = watered
        .iter()
        .filter(|&&key| {
            let record = settled.records.get(&key);
            !made(key)
                && !settled.done(key, false)
                && settled.missing.get(&key) != record
                && waiting(key)
        })
        .map(|&key| (nearness(key), key))
        .collect();
    fetching.sort_unstable();
    let cell = state.settled.cell;
    for (place, key) in making {
        let Some(&record) = state.settled.records.get(&key) else {
            continue;
        };
        let missing = state.settled.missing.get(&key) == Some(&record);
        if missing && !queue.settles {
            continue;
        }
        let priority = job_priority(VIEW_JOBS, place);
        if !slots.take(priority) {
            break;
        }
        let phase = if missing {
            // Settling from the water the tile holds under other edits,
            // where it holds any.
            let warm = state
                .settled
                .held
                .get(&key)
                .filter(|(_, held)| *held != record)
                .map(|(water, _)| (**water).clone());
            let (job, _) = state.water_record(stack, key, (cell, &queue.source), warm);
            Phase::Starting(slots.jobs.submit(WaterJob::KIND.name(), job.input(), priority))
        } else {
            Phase::Fetching(slots.jobs.fetch(record, priority))
        };
        let asked = Asked {
            phase,
            record,
            made: true,
        };
        state.settled.asked.insert(key, asked);
    }
    for (place, key) in fetching {
        let priority = job_priority(IDLE_JOBS, place);
        let Some(&record) = state.settled.records.get(&key) else {
            continue;
        };
        if !slots.take(priority) {
            break;
        }
        let asked = Asked {
            phase: Phase::Fetching(slots.jobs.fetch(record, priority)),
            record,
            made: false,
        };
        state.settled.asked.insert(key, asked);
    }
}

/// Takes what arrived of `state`'s settled water's records: records
/// fetched or settled on the GPU (through `queue`), held whole for the
/// resident level-0 tiles `near` and every one's summary; a record no host
/// holds counted missing (made next where its tile's is); and starts
/// prepared, handed to `queue` to settle.
fn take_arrived(
    state: &mut TerrainState,
    slots: &mut JobBudget<'_>,
    queue: &mut SettleQueue,
    near: &[TileKey],
    frame: u64,
) {
    let pyramid = state.terrain.pyramid;
    for (key, asked) in state
        .settled
        .asked
        .iter()
        .map(|(key, &asked)| (key, asked))
        .collect::<Vec<_>>()
    {
        let output = match asked.phase {
            Phase::Fetching(id) | Phase::Starting(id) => slots.jobs.take(id),
            Phase::Settling => queue.take(asked.record),
        };
        let Some(output) = output else {
            continue;
        };
        let settled = &mut state.settled;
        settled.asked.remove(&key);
        if let Phase::Starting(_) = asked.phase {
            match job_output(
                WaterJob::KIND.name(),
                output,
                engine_terrain::jobs::water_output,
            ) {
                Ok(start) if start.key == key && start.cell == settled.cell => {
                    queue.asked.push((asked.record, Arc::new(start)));
                    let settling = Asked {
                        phase: Phase::Settling,
                        ..asked
                    };
                    settled.asked.insert(key, settling);
                }
                Ok(_) => {
                    settled.failed.insert(key, frame);
                }
                Err(error) => {
                    settled.failed.insert(key, frame);
                    state.failures.record(frame, error);
                }
            }
            continue;
        }
        if let (Phase::Fetching(_), Err(error)) = (asked.phase, &output)
            && error.as_str() == MISSING_RECORD
        {
            settled.missing.insert(key, asked.record);
            continue;
        }
        match job_output(SETTLED_KIND, output, |bytes| {
            engine_terrain::codec::decode::<SteadyWater>(bytes)
        }) {
            Ok(water) if water.key == key && water.cell == settled.cell => {
                settled.failed.remove(&key);
                settled.missing.remove(&key);
                settled
                    .summaries
                    .insert(key, (water.summary.clone(), asked.record));
                if near.binary_search(&key).is_ok() {
                    settled.held.insert(key, (Arc::new(water), asked.record));
                }
                settled.arrive(&pyramid, key);
            }
            Ok(_) => {
                settled.failed.insert(key, frame);
            }
            Err(error) => {
                settled.failed.insert(key, frame);
                state.failures.record(frame, error);
            }
        }
    }
}

/// Gives `state`'s resident tiles the water their settled water stands:
/// the level-0 tiles of `near` whose records are held whole the water
/// their records and their neighbors' stand at their samples, coarser
/// tiles holding water the flow and foam of the summaries under them;
/// those given none yet, those that arrived anew since, and those whose
/// samples read a record that arrived since. Tiles whose rivers something
/// standing in them holds back keep the water those rivers run with.
fn give(state: &mut TerrainState, near: &[TileKey]) {
    let pyramid = state.terrain.pyramid;
    let resident = state
        .store
        .keys_where(|key| key.level > 0 || near.binary_search(key).is_ok());
    state
        .settled
        .given
        .retain(|key, _| resident.binary_search(key).is_ok());
    for key in resident {
        let latest = state.settled.changed.get(&key).copied().unwrap_or(0);
        if latest == 0
            || (key.level == 0 && !state.settled.held.contains_key(&key))
            || state.store.watered(&key).is_some_and(Option::is_some)
        {
            continue;
        }
        let Some(tile) = state.store.get(&key).cloned() else {
            continue;
        };
        if key.level > 0 && tile.water().is_none() {
            continue;
        }
        let current = state
            .settled
            .given
            .get(&key)
            .is_some_and(|(given, at)| Arc::ptr_eq(given, &tile) && *at >= latest);
        if current {
            continue;
        }
        let settled = &state.settled;
        let given = Arc::new(if key.level == 0 {
            tile.steadied(&pyramid, settled.cell, &|i, j| {
                settled.cell_at(&pyramid, i, j)
            })
        } else {
            tile.flowing(&pyramid, &|i, j| settled.square_at(i, j))
        });
        state.store.rewater(&key, Some(given.clone()), None);
        let arrived = state.settled.arrived;
        state.settled.given.insert(key, (given, arrived));
    }
}
