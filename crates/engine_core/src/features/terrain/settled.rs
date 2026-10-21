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
//! Each tile's record is the one its water settled to over its region of
//! the cook's, the terrain whole where it fits a region (else the fewest
//! regions that fit, [`engine_terrain::settle_regions`]); where no host
//! holds that one, outside a cooking session, the one it settled to over a
//! region of its own ([`SettleRegion::alone`]). The two are kept apart, so
//! a session that made a tile's water alone leaves the cook's to make. A
//! cooking session makes every record a region of the cook's keeps that no
//! host holds, together, once each of its tiles was fetched; another
//! session makes a tile's alone, from the record it holds under other
//! edits where it holds one.
//! Each level-0 tile a region reaches gives its part, a job on a worker
//! ([`WaterPart`], shared by the regions settling while it is made); the
//! parts make what the region settles from ([`SettleStart::assemble`]),
//! which the renderer settles on the GPU, handing each record to its host
//! to keep ([`SettleQueue`]); records the GPU made are cooked once, and a
//! verifying host compares those made again within their tolerance
//! ([`settled_alike`]).

use super::*;
use crate::{MISSING_RECORD, RecordParity};
use engine_terrain::jobs::WaterJob;
use engine_terrain::{
    SUMMARY_SIDE, SettleRegion, SettleStart, SettledKey, SteadyCell, SteadySquare, SteadySummary,
    SteadyWater, WaterPart,
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
/// packs and store (by its job), its region's parts being made, or
/// settling on the GPU.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Fetching(JobId),
    Starting,
    Settling,
}

/// A level-0 tile's records of settled water under the edits the ground
/// has now: over its region of the cook's, and alone.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Keys {
    region: RecordKey,
    alone: RecordKey,
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

/// A region whose tiles' records are made: the region, each tile it keeps'
/// record's key (in its order), whether it settles from the records its
/// tiles hold under other edits, the tiles whose parts it settles from,
/// and how near the view it is, which its parts' jobs go by.
struct Making {
    region: SettleRegion,
    records: Vec<RecordKey>,
    warm: bool,
    reached: Vec<TileKey>,
    place: u32,
}

/// A level-0 tile's part of the water settling over regions: its job asked
/// for, or made.
enum Part {
    Asked(JobId),
    Made(Arc<WaterPart>),
}

/// Water settling on the GPU, which terrains share: the hash of the
/// renderer's code settling it, which its records are keyed by with their
/// tiles' own ([`Terrains::set_water_settling`]), and whether a renderer
/// settles it here; the starts handed to the renderer, each with the keys
/// of the records it makes, until its host takes them; and what the
/// renderer settled, by key, for the frame's terrains to take (each a
/// record, or why it failed).
#[derive(Default)]
pub(super) struct SettleQueue {
    pub(super) source: [u8; 32],
    pub(super) settles: bool,
    asked: Vec<(Vec<RecordKey>, Arc<SettleStart>)>,
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
    /// call, each with the keys of the records it makes, in the order it
    /// keeps their tiles.
    pub fn take_water_settles(&mut self) -> Vec<(Vec<RecordKey>, Arc<SettleStart>)> {
        std::mem::take(&mut self.settling.asked)
    }

    /// The renderer's settled water of `record`: its record's bytes, or why
    /// it failed.
    pub fn water_settled(&mut self, record: RecordKey, settled: Result<Vec<u8>, String>) {
        self.settling.settled.push((record, settled));
    }
}

/// A terrain's settled water: the cells' size its records hold; the
/// level-0 tiles its bake's water reaches, found over its baked layers,
/// and the cook's regions over them (each a rectangle of tiles, with its
/// watered ones); the generation of tiles its records' keys were found
/// for, and each watered tile's records' keys then; the records of
/// resident level-0 tiles held whole and every record's summary, each with
/// its key; the records asked for, those that failed (at the frame they
/// did) and those no host holds; the regions whose records are made and
/// the parts they settle from; per tile, the latest of the records its
/// samples read to arrive, arrivals counted; the resident tiles given their
/// water, as given and when; and whether its session cooks, which makes
/// every watered tile's over the cook's regions.
#[derive(Default)]
pub(super) struct Settled {
    cell: f32,
    watered: Option<(Arc<Layers>, Arc<[TileKey]>)>,
    regions: Arc<[([u32; 4], Vec<TileKey>)]>,
    checked: Option<Generation>,
    records: TileMap<Keys>,
    pub(super) held: TileMap<(Arc<SteadyWater>, RecordKey)>,
    summaries: TileMap<(SteadySummary, RecordKey)>,
    asked: TileMap<Asked>,
    failed: TileMap<u64>,
    missing: HashSet<RecordKey>,
    making: Vec<Making>,
    parts: TileMap<Part>,
    changed: TileMap<u64>,
    given: TileMap<(Arc<Tile>, u64)>,
    arrived: u64,
    pub(super) cooking: bool,
}

impl Settled {
    /// The jobs of its records and parts asked for.
    pub(super) fn asked(&self) -> Vec<JobId> {
        let fetches = self
            .asked
            .iter()
            .filter_map(|(_, asked)| match asked.phase {
                Phase::Fetching(id) => Some(id),
                Phase::Starting | Phase::Settling => None,
            });
        let parts = self.parts.iter().filter_map(|(_, part)| match part {
            Part::Asked(id) => Some(*id),
            Part::Made(_) => None,
        });
        fetches.chain(parts).collect()
    }

    /// Records arrived, whose summaries it holds, and those asked for.
    pub(super) fn counts(&self) -> (u32, u32) {
        (self.summaries.len() as u32, self.asked.len() as u32)
    }

    /// Watered tiles whose record over their region of the cook's under
    /// the edits the ground has now has not arrived and did not fail: what
    /// a cook has left.
    pub(super) fn left(&self) -> u64 {
        let Some((_, watered)) = &self.watered else {
            return 0;
        };
        watered
            .iter()
            .filter(|&&key| {
                let region = self.records.get(&key).map(|keys| keys.region);
                self.summaries.get(&key).map(|(_, record)| *record) != region
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

    /// The record level-0 tile `key` wants under the edits the ground has
    /// now: over its region of the cook's, or, outside a cooking session
    /// where no host holds that one, alone.
    fn wanted(&self, key: TileKey) -> Option<RecordKey> {
        let keys = self.records.get(&key)?;
        if self.cooking || !self.missing.contains(&keys.region) {
            Some(keys.region)
        } else {
            Some(keys.alone)
        }
    }

    /// Whether level-0 tile `key`'s wanted record arrived as a tile `near`
    /// keeps it: held whole, or for another tile its summary.
    fn done(&self, key: TileKey, near: bool) -> bool {
        let wanted = self.wanted(key);
        if near {
            self.held.get(&key).map(|(_, held)| *held) == wanted
        } else {
            self.summaries.get(&key).map(|(_, held)| *held) == wanted
        }
    }

    /// Whether no host holds level-0 tile `key`'s record over its region of
    /// the cook's under the edits the ground has now.
    fn missing(&self, key: TileKey) -> bool {
        self.records
            .get(&key)
            .is_some_and(|keys| self.missing.contains(&keys.region))
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

    /// Fails the records of `keys`, at `frame`: asked for no more, and
    /// asked for again a while after.
    fn fail(&mut self, keys: &[TileKey], frame: u64) {
        for &key in keys {
            self.asked.remove(&key);
            self.failed.insert(key, frame);
        }
    }
}

impl TerrainState {
    /// The keys of level-0 tile `key`'s settled water's records over the
    /// stack shared as `stack`, under the edits reaching the square it
    /// settles over alone, in cells `cell` meters across, settled by code
    /// hashing to `settling`: over the region of `tiles` (its region of the
    /// cook's), and alone.
    fn water_records(
        &self,
        stack: &str,
        (key, tiles): (TileKey, [u32; 4]),
        (cell, settling): (f32, &[u8; 32]),
    ) -> Keys {
        let pyramid = self.terrain.pyramid;
        let (low, high) = SettleRegion::alone(&pyramid, key, cell).bounds();
        let mut settled = SettledKey {
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
            tiles,
        };
        let record = |settled: &SettledKey| {
            let identity = [settling.as_slice(), &settled.identity()].concat();
            RecordKey::new(&engine_terrain::sources::WATER, SETTLED_KIND, &identity)
        };
        let region = record(&settled);
        settled.tiles = [u32::from(key.x), u32::from(key.z), 1, 1];
        Keys {
            region,
            alone: record(&settled),
        }
    }

    /// The job making level-0 tile `key`'s part of the water settling over
    /// the stack shared as `stack`, under the edits reaching its ground, in
    /// cells `cell` meters across.
    fn water_part(&self, stack: &str, key: TileKey, cell: f32) -> WaterJob {
        let pyramid = self.terrain.pyramid;
        WaterJob {
            stack: stack.to_owned(),
            pyramid,
            key,
            stamps: self
                .terrain
                .edits
                .iter()
                .map(|edit| edit.stamp)
                .filter(|stamp| reaches(stamp, &pyramid, key))
                .collect(),
            cell,
        }
    }
}

/// Cancels every record and part `state`'s settled water asked for, giving
/// `slots` theirs back.
fn cancel_all(state: &TerrainState, slots: &mut JobBudget<'_>) {
    for id in state.settled.asked() {
        slots.jobs.cancel(id);
        slots.release();
    }
}

/// Follows `state`'s settled water for this frame, where `water` asks for
/// water: takes the records that arrived (fetched, or settled on the GPU
/// through `queue`) and the parts made, handing `queue` the regions whose
/// parts are all made; asks for the records of the resident level-0 tiles
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
    let leaving = settled
        .asked
        .keys_where(|key, asked| asked.made && !cooking && near.binary_search(key).is_err());
    for key in leaving {
        if let Some(asked) = settled.asked.remove(&key) {
            match asked.phase {
                Phase::Fetching(id) => {
                    slots.jobs.cancel(id);
                    slots.release();
                }
                // What its region settles still comes, for its host to
                // keep; a region none of whose tiles is asked for stops.
                Phase::Starting | Phase::Settling => {}
            }
        }
    }
    ask(state, slots, queue, (&stack, &watered, &near), (eye, frame));
    release_parts(&mut state.settled, slots);
    give(state, &near);
}

/// Readies `state`'s settled water for `water`: none without water or a
/// bake's water (what it asked for cancelled, `slots` given theirs back),
/// anew over other baked layers or cells' size, and each watered tile's
/// record's key found again under other edits (keyed by `queue`'s
/// settling), the records asked for under others asked for no more. The
/// stack its records settle over and the level-0 tiles its bake's water
/// reaches, in key order.
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
        let pyramid = state.terrain.pyramid;
        let watered: Arc<[TileKey]> = match state.settled.watered.take() {
            Some((over, keys)) if Arc::ptr_eq(&over, &baked) => keys,
            _ => engine_terrain::watered_tiles(&baked, &pyramid).into(),
        };
        let cell = water.simulation_cell;
        let regions = engine_terrain::settle_regions(&pyramid, cell)
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
        let cooking = state.settled.cooking;
        state.settled = Settled {
            cell,
            watered: Some((baked, watered)),
            regions,
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
        for (tiles, keys) in state.settled.regions.iter() {
            for &key in keys {
                let found = state.water_records(&stack, (key, *tiles), (cell, &queue.source));
                records.insert(key, found);
            }
        }
        let settled = &mut state.settled;
        settled.records = records;
        settled.checked = Some(generation);
        let records = &settled.records;
        let stale = settled.asked.keys_where(|key, asked| {
            records
                .get(key)
                .is_none_or(|keys| asked.record != keys.region && asked.record != keys.alone)
        });
        for key in stale {
            if let Some(Asked {
                phase: Phase::Fetching(id),
                ..
            }) = settled.asked.remove(&key)
            {
                slots.jobs.cancel(id);
                slots.release();
            }
        }
    }
    Some((stack, watered))
}

/// Asks, while `slots` admits them, for the records over the stack shared
/// as `stack` that `state`'s settled water lacks under the edits the ground
/// has now, nearest `eye` (terrain-local) first: those of the resident
/// level-0 tiles `near` (every tile of `watered` while its session cooks),
/// fetched where a host holds them and else made over regions where
/// `queue` has a renderer settle them (each tile alone, from the record it
/// holds under other edits where it holds any; while its session cooks,
/// those of each of the cook's regions together once each of its tiles was
/// fetched), the parts the regions settle from asked for; then those of
/// the other tiles of `watered`, fetched where a host holds them. A record
/// that failed is asked for again a while after it did, `frame` now.
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
    // What a tile asks a host for: its record over its region of the
    // cook's, else (outside a cooking session) alone, unless no host holds
    // either.
    let fetched = |key: TileKey| {
        let keys = settled.records.get(&key)?;
        if !settled.missing.contains(&keys.region) {
            Some(keys.region)
        } else if cooking || settled.missing.contains(&keys.alone) {
            None
        } else {
            Some(keys.alone)
        }
    };
    let mut fetching: Vec<(u32, TileKey, RecordKey)> = watered
        .iter()
        .filter(|&&key| !made(key) && !settled.done(key, false) && waiting(key))
        .filter_map(|&key| Some((nearness(key), key, fetched(key)?)))
        .collect();
    fetching.sort_unstable_by_key(|&(place, key, _)| (place, key));
    let making: Vec<(u32, TileKey, Option<RecordKey>)> = making
        .into_iter()
        .map(|(place, key)| (place, key, fetched(key)))
        .collect();
    let cell = settled.cell;
    let mut alone = Vec::new();
    for (place, key, record) in making {
        // No host holds the record it wants: its region of the cook's makes
        // it while a session cooks, else it settles alone.
        let Some(record) = record else {
            if queue.settles && !cooking && state.settled.records.contains_key(&key) {
                alone.push((place, key));
            }
            continue;
        };
        let priority = job_priority(VIEW_JOBS, place);
        if !slots.take(priority) {
            break;
        }
        let asked = Asked {
            phase: Phase::Fetching(slots.jobs.fetch(record, priority)),
            record,
            made: true,
        };
        state.settled.asked.insert(key, asked);
    }
    if queue.settles {
        let regions: Vec<(SettleRegion, bool, u32)> = if cooking {
            cook_regions(state)
        } else {
            alone
                .into_iter()
                .map(|(place, key)| {
                    let alone = state.settled.records.get(&key).map(|keys| keys.alone);
                    let warm = state
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
            start_making(&mut state.settled, &pyramid, region, alone, (warm, place));
        }
        ask_parts(state, stack, slots);
    }
    for (place, key, record) in fetching {
        let priority = job_priority(IDLE_JOBS, place);
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

/// The cook's regions of `state`'s settled water ready to settle, each
/// over its rectangle of tiles (in cells of its records' size), keeping
/// those of its watered tiles no host holds the records of, neither asked
/// for nor failed, once each of them was fetched: not warmed, and all
/// alike near.
fn cook_regions(state: &TerrainState) -> Vec<(SettleRegion, bool, u32)> {
    let pyramid = state.terrain.pyramid;
    let settled = &state.settled;
    let cell = settled.cell;
    let mut ready = Vec::new();
    for (rectangle, tiles) in settled.regions.iter() {
        let fetched = tiles.iter().all(|&key| {
            let fetching = matches!(
                settled.asked.get(&key),
                Some(Asked {
                    phase: Phase::Fetching(_),
                    ..
                })
            );
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

/// Has `settled` make `region`'s records (those its tiles settle to
/// `alone`, else those over their regions of the cook's), from the records
/// its tiles hold under other edits where `warm`, its parts' jobs going by
/// `place`.
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
        records.push(if alone { keys.alone } else { keys.region });
    }
    for (&key, &record) in region.kept.iter().zip(&records) {
        let asked = Asked {
            phase: Phase::Starting,
            record,
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

/// Asks, while `slots` admits them, for the parts over the stack shared as
/// `stack` the regions `state`'s settled water makes settle from and lack,
/// nearest region first.
fn ask_parts(state: &mut TerrainState, stack: &str, slots: &mut JobBudget<'_>) {
    let settled = &state.settled;
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
        if state.settled.parts.contains_key(&key) {
            continue;
        }
        let priority = job_priority(VIEW_JOBS, place);
        if !slots.take(priority) {
            break;
        }
        let job = state.water_part(stack, key, cell);
        let id = slots
            .jobs
            .submit(WaterJob::KIND.name(), job.input(), priority);
        state.settled.parts.insert(key, Part::Asked(id));
    }
}

/// Drops the regions `settled` makes none of whose tiles it asks for any
/// more, and the parts no region it makes settles from, cancelling those
/// asked for (`slots` given theirs back).
fn release_parts(settled: &mut Settled, slots: &mut JobBudget<'_>) {
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
                    .is_some_and(|asked| asked.phase == Phase::Starting && asked.record == *record)
            })
    });
    let making = &settled.making;
    settled.parts.retain(|key, part| {
        let wanted = making
            .iter()
            .any(|making| making.reached.binary_search(key).is_ok());
        if !wanted && let Part::Asked(id) = part {
            slots.jobs.cancel(*id);
            slots.release();
        }
        wanted
    });
}

/// Takes what arrived of `state`'s settled water's records and parts:
/// records fetched or settled on the GPU (through `queue`), held whole for
/// the resident level-0 tiles `near` and every one's summary; a record no
/// host holds counted missing (made next where its tile's is); parts made,
/// and regions whose parts are all made assembled and handed to `queue` to
/// settle (a region that fails failing its tiles' records, at `frame`).
fn take_arrived(
    state: &mut TerrainState,
    slots: &mut JobBudget<'_>,
    queue: &mut SettleQueue,
    near: &[TileKey],
    frame: u64,
) {
    let pyramid = state.terrain.pyramid;
    take_parts(state, slots, queue, frame);
    for (key, asked) in state
        .settled
        .asked
        .iter()
        .map(|(key, &asked)| (key, asked))
        .collect::<Vec<_>>()
    {
        let output = match asked.phase {
            Phase::Fetching(id) => slots.jobs.take(id),
            Phase::Settling => queue.take(asked.record),
            Phase::Starting => None,
        };
        let Some(output) = output else {
            continue;
        };
        let settled = &mut state.settled;
        settled.asked.remove(&key);
        if let (Phase::Fetching(_), Err(error)) = (asked.phase, &output)
            && error.as_str() == MISSING_RECORD
        {
            settled.missing.insert(asked.record);
            continue;
        }
        match job_output(SETTLED_KIND, output, |bytes| {
            engine_terrain::codec::decode::<SteadyWater>(bytes)
        }) {
            Ok(water) if water.key == key && water.cell == settled.cell => {
                settled.failed.remove(&key);
                settled.missing.remove(&asked.record);
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

/// Takes the parts `state`'s settled water asked for that were made, a
/// part that failed failing the regions settling from it (at `frame`); and
/// assembles each region whose parts are all made, handing `queue` what it
/// settles from and the keys of its records.
fn take_parts(
    state: &mut TerrainState,
    slots: &mut JobBudget<'_>,
    queue: &mut SettleQueue,
    frame: u64,
) {
    let settled = &mut state.settled;
    let mut broken = Vec::new();
    for key in settled.parts.keys() {
        let Some(Part::Asked(id)) = settled.parts.get(&key) else {
            continue;
        };
        let Some(output) = slots.jobs.take(*id) else {
            continue;
        };
        match job_output(
            WaterJob::KIND.name(),
            output,
            engine_terrain::jobs::water_output,
        ) {
            Ok(part) if part.key == key && part.cell == settled.cell => {
                settled.parts.insert(key, Part::Made(Arc::new(part)));
            }
            Ok(_) => {
                settled.parts.remove(&key);
                broken.push(key);
            }
            Err(error) => {
                settled.parts.remove(&key);
                broken.push(key);
                state.failures.record(frame, error);
            }
        }
    }
    let pyramid = state.terrain.pyramid;
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
                for (&key, &record) in making.region.kept.iter().zip(&making.records) {
                    let starting = settled.asked.get(&key).is_some_and(|asked| {
                        asked.phase == Phase::Starting && asked.record == record
                    });
                    if starting {
                        let asked = Asked {
                            phase: Phase::Settling,
                            record,
                            made: true,
                        };
                        settled.asked.insert(key, asked);
                    }
                }
                queue.asked.push((making.records, Arc::new(start)));
            }
            Err(error) => {
                settled.fail(&making.region.kept, frame);
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
