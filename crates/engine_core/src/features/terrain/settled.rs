//! Terrains' settled water: each level-0 tile's rivers at their steady
//! state ([`SteadyWater`]), records the cook makes for every tile its
//! bake's water reaches and hosts keep, at the cells' size water's
//! simulation takes ([`Water::simulation_cell`]). Frames ask for the
//! records of the resident level-0 tiles water reaches and of the watered
//! tiles beside them, held whole, and fetch every other tile's where a host
//! holds it, nearest first, for its summary ([`engine_terrain::SteadySummary`]).
//! Each record's head (its summary) is shared with workers as it arrives,
//! and the records held whole are shared whole.
//!
//! A tile's water is a record of its own, a job on a worker
//! ([`SteadiedJob`]) reading the tile's record and the shared records
//! around it: a resident level-0 tile's water stands as its record and its
//! neighbors' say ([`engine_terrain::Tile::steadied`]), a resident coarser
//! tile's runs and foams as the summaries under it and around it say
//! ([`engine_terrain::Tile::flowing`]). It is asked for once everything it
//! reads is here, or known never to come, and kept by the keys of what it
//! reads, so it is made again only when one of them changes, as after an
//! edit, never as each neighbor arrives; the cook ships them as every
//! record. Frames only swap what the jobs made into the resident tiles:
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
//! host holds, together, once each of its tiles was fetched, and every
//! watered tile's water and every coarser tile's over them; another
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
use engine_terrain::codec::CodecError;
use engine_terrain::jobs::{
    SteadiedJob, WaterJob, settled_key, share_settled, share_summary, summary_key,
};
use engine_terrain::{
    SettleRegion, SettleStart, SettledKey, SteadyCell, SteadyWater, TileWater, WaterPart,
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
/// whether it is made where no host holds it (or only fetched).
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

/// What a tile's water reads: its tile's record and the records of settled
/// water of the level-0 tiles it reads (whole for a level-0 tile, their
/// summaries for a coarser one), each by its tile with its key, in key
/// order.
#[derive(Clone, PartialEq, Eq)]
struct Reads {
    tile: RecordKey,
    settled: Vec<(TileKey, RecordKey)>,
}

/// What a tile's water reads as its settled water stands: all of it here,
/// some still to come, or nothing to read.
enum Wants {
    Here(Reads),
    Coming,
    Nothing,
}

/// Where a record of settled water a tile's water reads stands: here, with
/// its key; still to come; or never coming (it failed, or no host holds it
/// and nothing makes it).
enum Found {
    Here(RecordKey),
    Coming,
    Never,
}

/// A tile's water from settled water: what it reads and its record's key;
/// its job while asked for; and once made, the water (none where none
/// stands on the tile), kept for tiles resident when it came.
struct Watering {
    reads: Reads,
    record: RecordKey,
    job: Option<JobId>,
    made: Option<Arc<Option<TileWater>>>,
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
/// for, and each watered tile's records' keys then; the records held whole,
/// decoded, and those shared whole with workers, and the record of each
/// tile whose head (its summary) is shared, each by its key; the records
/// asked for, those that failed (at the frame they did) and those no host
/// holds; the regions whose records are made and the parts they settle
/// from; the tiles' jobs and records' keys its tiles' water reads, over the
/// recipe and generation they were found for; each tile's water asked for
/// or made, those whose jobs failed (at the frame they did), and the tiles
/// given their water, as given and with its key; the keys it shares with
/// workers; how often what its tiles' water reads changed, and what it last
/// asked for their water at, whether it waits to ask again, and how many
/// tiles' water its cooking session has left; and whether its session
/// cooks, which makes every watered tile's records over the cook's regions
/// and every tile's water over them.
#[derive(Default)]
pub(super) struct Settled {
    cell: f32,
    watered: Option<(Arc<Layers>, Arc<[TileKey]>)>,
    regions: Arc<[([u32; 4], Vec<TileKey>)]>,
    checked: Option<Generation>,
    records: TileMap<Keys>,
    pub(super) held: TileMap<(Arc<SteadyWater>, RecordKey)>,
    shared: TileMap<RecordKey>,
    summaries: TileMap<RecordKey>,
    asked: TileMap<Asked>,
    failed: TileMap<u64>,
    missing: HashSet<RecordKey>,
    making: Vec<Making>,
    parts: TileMap<Part>,
    tiles: TileMap<(TileJob, RecordKey)>,
    tiles_for: Option<(String, Generation)>,
    waters: TileMap<Watering>,
    waters_failed: TileMap<u64>,
    given: TileMap<(Arc<Tile>, RecordKey)>,
    sharing: HashSet<String>,
    revision: u64,
    seen: Option<(u64, u64, Generation, bool)>,
    deferred: bool,
    waters_left: u64,
    pub(super) cooking: bool,
}

impl Settled {
    /// The jobs of its records, parts and tiles' water asked for.
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
        let waters = self.waters.iter().filter_map(|(_, watering)| watering.job);
        fetches.chain(parts).chain(waters).collect()
    }

    /// Records arrived, whose summaries it shares, and those asked for.
    pub(super) fn counts(&self) -> (u32, u32) {
        (self.summaries.len() as u32, self.asked.len() as u32)
    }

    /// Watered tiles whose record over their region of the cook's under
    /// the edits the ground has now has not arrived and did not fail, and
    /// the tiles whose water its cooking session has still to make: what a
    /// cook has left.
    pub(super) fn left(&self) -> u64 {
        let Some((_, watered)) = &self.watered else {
            return 0;
        };
        let records = watered
            .iter()
            .filter(|&&key| {
                let region = self.records.get(&key).map(|keys| keys.region);
                self.summaries.get(&key).copied() != region && !self.failed.contains_key(&key)
            })
            .count() as u64;
        records + self.waters_left
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

    /// The keys it shares with workers, in order.
    pub(super) fn shares(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.sharing.iter().cloned().collect();
        keys.sort_unstable();
        keys
    }

    /// Whether it shares `key` with workers.
    pub(super) fn holds_share(&self, key: &str) -> bool {
        self.sharing.contains(key)
    }

    /// Shares the bytes `bytes` makes with workers as `key`, unless it does
    /// already; another terrain sharing the same takes it up.
    fn share(&mut self, jobs: &mut Jobs, key: String, bytes: impl FnOnce() -> Vec<u8>) {
        if !self.sharing.insert(key.clone()) {
            return;
        }
        if !jobs.is_shared(&key) {
            jobs.share(key, bytes());
        }
    }

    /// Lets go of its share `key`: `released` takes it, for streaming to
    /// drop at the frame's end where no terrain holds it.
    fn release(&mut self, key: String, released: &mut Vec<String>) {
        if self.sharing.remove(&key) {
            released.push(key);
        }
    }

    /// Lets go of every share it holds into `released`.
    fn release_all(&mut self, released: &mut Vec<String>) {
        released.extend(self.shares());
        self.sharing.clear();
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

    /// The record level-0 tile `key` asks a host for: its record over its
    /// region of the cook's, else (outside a cooking session) alone, unless
    /// no host holds either.
    fn fetched(&self, key: TileKey) -> Option<RecordKey> {
        let keys = self.records.get(&key)?;
        if !self.missing.contains(&keys.region) {
            Some(keys.region)
        } else if self.cooking || self.missing.contains(&keys.alone) {
            None
        } else {
            Some(keys.alone)
        }
    }

    /// Whether level-0 tile `key`'s wanted record arrived as a tile kept
    /// `whole` takes it: shared whole, or for another tile its summary.
    fn done(&self, key: TileKey, whole: bool) -> bool {
        let here = if whole {
            self.shared.get(&key)
        } else {
            self.summaries.get(&key)
        };
        here.copied() == self.wanted(key)
    }

    /// Whether no host holds level-0 tile `key`'s record over its region of
    /// the cook's under the edits the ground has now.
    fn missing(&self, key: TileKey) -> bool {
        self.records
            .get(&key)
            .is_some_and(|keys| self.missing.contains(&keys.region))
    }

    /// Where the record of level-0 tile `key` a tile's water reads stands,
    /// shared whole (`whole`) or its head: here; still to come, asked for,
    /// to fetch, or to make (`made` where it is made, `settles` where a
    /// renderer settles water); or never coming.
    fn found(&self, key: TileKey, whole: bool, (made, settles): (bool, bool)) -> Found {
        let here = if whole {
            self.shared.get(&key)
        } else {
            self.summaries.get(&key)
        };
        if let Some(&record) = here {
            return Found::Here(record);
        }
        let coming = !self.failed.contains_key(&key)
            && (self.asked.contains_key(&key)
                || self.fetched(key).is_some()
                || (made && settles && self.records.contains_key(&key)));
        if coming { Found::Coming } else { Found::Never }
    }

    /// Fails the records of `keys`, at `frame`: asked for no more, and
    /// asked for again a while after.
    fn fail(&mut self, keys: &[TileKey], frame: u64) {
        for &key in keys {
            self.asked.remove(&key);
            self.failed.insert(key, frame);
        }
        self.revision += 1;
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

    /// Tile `key`'s job over the recipe shared as `recipe` and its record's
    /// key, as its settled water found them under the generation tiles are
    /// made from now.
    fn settled_tile(&mut self, recipe: &str, key: TileKey) -> (TileJob, RecordKey) {
        let basis = (recipe.to_owned(), self.store.generation());
        if self.settled.tiles_for.as_ref() != Some(&basis) {
            self.settled.tiles = TileMap::default();
            self.settled.tiles_for = Some(basis);
        }
        if let Some(found) = self.settled.tiles.get(&key) {
            return found.clone();
        }
        let found = self.tile_record(recipe, key);
        self.settled.tiles.insert(key, found.clone());
        found
    }
}

/// Cancels every record, part and tile's water `state`'s settled water
/// asked for, giving `slots` theirs back.
fn cancel_all(state: &TerrainState, slots: &mut JobBudget<'_>) {
    for id in state.settled.asked() {
        slots.jobs.cancel(id);
        slots.release();
    }
}

/// Follows `state`'s settled water for this frame, where `water` asks for
/// water: takes the records that arrived (fetched, or settled on the GPU
/// through `queue`), sharing them with workers, and the parts made, handing
/// `queue` the regions whose parts are all made; asks for the records of
/// the resident level-0 tiles its bake's water reaches and of the watered
/// tiles beside them, whole (every watered tile's while its session cooks),
/// and fetches the other watered tiles' while `slots` admits them, nearest
/// `eye` (terrain-local) first, those of tiles no longer resident made no
/// more; and has its tiles' water made on workers and swapped into resident
/// tiles within `work`. The shares it lets go of go to `released`. Without
/// water, or without a bake's water, it holds nothing.
pub(super) fn follow(
    state: &mut TerrainState,
    water: Option<&Water>,
    (slots, work): (&mut JobBudget<'_>, &Work),
    (queue, released): (&mut SettleQueue, &mut Vec<String>),
    (eye, frame): (Vec2, u64),
) {
    let Some((stack, watered)) = refresh(state, water, slots, queue, released) else {
        return;
    };
    let pyramid = state.terrain.pyramid;
    let near: Vec<TileKey> = state
        .store
        .keys_where(|key| key.level == 0 && watered.binary_search(key).is_ok());
    // Held whole: the tiles of `near` and the watered tiles beside them,
    // which their water reads; shared whole while a session cooks, every
    // watered tile's.
    let kept = beside(&pyramid, &near, &watered);
    let whole: Vec<TileKey> = if state.settled.cooking {
        watered.to_vec()
    } else {
        kept.clone()
    };
    take_arrived(state, slots, (queue, released), (&kept, &whole), frame);
    let settled = &mut state.settled;
    let held = settled.held.len();
    settled
        .held
        .retain(|key, _| kept.binary_search(key).is_ok());
    let mut gone = Vec::new();
    settled.shared.retain(|key, record| {
        let keep = whole.binary_search(key).is_ok();
        if !keep {
            gone.push(settled_key(&record.0));
        }
        keep
    });
    if held != settled.held.len() || !gone.is_empty() {
        settled.revision += 1;
    }
    for key in gone {
        settled.release(key, released);
    }
    let cooking = settled.cooking;
    let leaving = settled
        .asked
        .keys_where(|key, asked| asked.made && !cooking && near.binary_search(key).is_err());
    for key in leaving {
        if let Some(asked) = settled.asked.remove(&key) {
            settled.revision += 1;
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
    ask(
        state,
        slots,
        queue,
        (&stack, &watered, &near, &whole),
        (eye, frame),
    );
    release_parts(&mut state.settled, slots);
    water_tiles(
        state,
        (slots, work),
        (&watered, &near),
        (eye, frame, queue.settles),
    );
}

/// The tiles of `watered` (in key order) among `near`, level-0 tiles of
/// `pyramid`, or beside one of them, in key order.
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

/// Readies `state`'s settled water for `water`: none without water or a
/// bake's water (what it asked for cancelled, `slots` given theirs back,
/// its shares let go of into `released`), anew over other baked layers or
/// cells' size, and each watered tile's record's key found again under
/// other edits (keyed by `queue`'s settling), the records asked for under
/// others asked for no more. The stack its records settle over and the
/// level-0 tiles its bake's water reaches, in key order.
fn refresh(
    state: &mut TerrainState,
    water: Option<&Water>,
    slots: &mut JobBudget<'_>,
    queue: &SettleQueue,
    released: &mut Vec<String>,
) -> Option<(String, Arc<[TileKey]>)> {
    let asking = water.filter(|water| water.enabled && state.around.is_none());
    let (Some(water), Some(stack), Some(baked)) = (
        asking,
        state.recipe.as_ref().map(|recipe| recipe.stack_key.clone()),
        state.baked.clone(),
    ) else {
        cancel_all(state, slots);
        state.settled.release_all(released);
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
        state.settled.release_all(released);
        let pyramid = state.terrain.pyramid;
        let watered: Arc<[TileKey]> = match state.settled.watered.take() {
            Some((over, keys)) if Arc::ptr_eq(&over, &baked) => keys,
            _ => engine_terrain::watered_tiles(&baked, &pyramid).into(),
        };
        let cell = water.simulation_cell;
        let regions = engine_terrain::settle_regions(&baked, &pyramid, cell)
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
        settled.revision += 1;
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
/// fetched), the parts the regions settle from asked for; with them those
/// of the other tiles of `whole`, fetched whole where a host holds them;
/// then those of the other tiles of `watered`, fetched where a host holds
/// them, for their summaries. A record that failed is asked for again a
/// while after it did, `frame` now.
fn ask(
    state: &mut TerrainState,
    slots: &mut JobBudget<'_>,
    queue: &SettleQueue,
    (stack, watered, near, whole): (&str, &[TileKey], &[TileKey], &[TileKey]),
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
    let mut fetching: Vec<(u32, TileKey, RecordKey)> = watered
        .iter()
        .filter(|&&key| !made(key) && !is_whole(key) && !settled.done(key, false) && waiting(key))
        .filter_map(|&key| Some((nearness(key), key, settled.fetched(key)?)))
        .collect();
    fetching.sort_unstable_by_key(|&(place, key, _)| (place, key));
    let making: Vec<(u32, TileKey, Option<RecordKey>)> = making
        .into_iter()
        .map(|(place, key)| (place, key, settled.fetched(key)))
        .collect();
    let cell = settled.cell;
    let mut alone = Vec::new();
    for (place, key, record) in making {
        // No host holds the record it wants: its region of the cook's makes
        // it while a session cooks, else it settles alone where it is near.
        let Some(record) = record else {
            if queue.settles && !cooking && is_near(key) && state.settled.records.contains_key(&key)
            {
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
            made: made(key),
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
/// records fetched or settled on the GPU (through `queue`), each taken in
/// ([`arrive`]) for the tiles `kept` holds whole and `whole` shares whole;
/// a record no host holds counted missing (made next where its tile's is);
/// parts made, and regions whose parts are all made assembled and handed to
/// `queue` to settle (a region that fails failing its tiles' records, at
/// `frame`). The shares it lets go of go to `released`.
fn take_arrived(
    state: &mut TerrainState,
    slots: &mut JobBudget<'_>,
    (queue, released): (&mut SettleQueue, &mut Vec<String>),
    (kept, whole): (&[TileKey], &[TileKey]),
    frame: u64,
) {
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
        settled.revision += 1;
        let bytes = match output {
            Ok(bytes) => bytes,
            Err(error)
                if matches!(asked.phase, Phase::Fetching(_)) && error == MISSING_RECORD =>
            {
                settled.missing.insert(asked.record);
                continue;
            }
            Err(error) => {
                settled.failed.insert(key, frame);
                state
                    .failures
                    .record(frame, format!("{SETTLED_KIND}: {error}"));
                continue;
            }
        };
        match arrive(
            state,
            slots.jobs,
            (key, asked.record, bytes),
            (kept, whole),
            released,
        ) {
            Ok(true) => {}
            Ok(false) => {
                state.settled.failed.insert(key, frame);
            }
            Err(error) => {
                state.settled.failed.insert(key, frame);
                state.failures.record(frame, error);
            }
        }
    }
}

/// Takes `bytes`, the record `record` of level-0 tile `key`'s settled
/// water, into `state`'s settled water, in place of the tile's record
/// before (whose shares `released` takes): its head (its summary) shared
/// with workers through `jobs`, for coarser tiles' water; the record
/// shared whole where `whole` holds the tile, for level-0 tiles' water; and
/// held whole where `kept` does, read here for the window simulated around
/// the camera and the regions settling warm. Whether it is the tile's, on
/// the lattice of its records' cells; an error where it does not read.
fn arrive(
    state: &mut TerrainState,
    jobs: &mut Jobs,
    (key, record, bytes): (TileKey, RecordKey, Vec<u8>),
    (kept, whole): (&[TileKey], &[TileKey]),
    released: &mut Vec<String>,
) -> Result<bool, String> {
    let settled = &mut state.settled;
    let unread = |error: CodecError| format!("{SETTLED_KIND}: its output does not decode: {error}");
    let (head, length) = SteadyWater::head(&bytes).map_err(unread)?;
    if head.key != key || head.cell != settled.cell {
        return Ok(false);
    }
    let held = if kept.binary_search(&key).is_ok() {
        Some(Arc::new(
            engine_terrain::codec::decode::<SteadyWater>(&bytes).map_err(unread)?,
        ))
    } else {
        None
    };
    settled.failed.remove(&key);
    settled.missing.remove(&record);
    if let Some(before) = settled.summaries.insert(key, record)
        && before != record
    {
        settled.release(summary_key(&before.0), released);
    }
    settled.share(jobs, summary_key(&record.0), || {
        share_summary(&bytes[..length])
    });
    if whole.binary_search(&key).is_ok() {
        if let Some(before) = settled.shared.insert(key, record)
            && before != record
        {
            settled.release(settled_key(&before.0), released);
        }
        settled.share(jobs, settled_key(&record.0), || share_settled(&bytes));
    }
    if let Some(water) = held {
        settled.held.insert(key, (water, record));
    }
    Ok(true)
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

/// Gives `state`'s tiles the water their settled water stands, each tile's
/// a job's record ([`SteadiedJob`]) over the recipe its tiles generate
/// from: takes what jobs made, within `work`'s time; once what its records
/// or resident tiles hold changed (or asking had to wait), asks for each
/// tile's water whose reads are all here and differ from those asked for,
/// as `slots` admits them, nearest `eye` (terrain-local) first, a tile whose
/// reads changed no longer waiting for what it asked for before, and the
/// water of tiles it no longer gives water stopping; then swaps what was
/// made into the resident tiles not holding it, within `work`'s time and
/// uploads. `watered` are the level-0 tiles its bake's water reaches, of
/// which `near` are resident; a record no host holds is made where
/// `settles` and its tile is near (or its session cooks); a job that
/// failed is asked for again a while after it did, `frame` now.
fn water_tiles(
    state: &mut TerrainState,
    (slots, work): (&mut JobBudget<'_>, &Work),
    (watered, near): (&[TileKey], &[TileKey]),
    (eye, frame, settles): (Vec2, u64, bool),
) {
    let Some(recipe) = state.recipe.as_ref().map(|recipe| recipe.key.clone()) else {
        return;
    };
    take_waters(state, slots, work, frame);
    let seen = (
        state.settled.revision,
        state.store.revision(),
        state.store.generation(),
        state.settled.cooking,
    );
    if state.settled.seen != Some(seen) || state.settled.deferred {
        state.settled.seen = Some(seen);
        ask_waters(
            state,
            slots,
            (&recipe, watered, near),
            (eye, frame, settles),
        );
    }
    give_waters(state, work);
}

/// Takes the water of `state`'s tiles its jobs made, within `work`'s time
/// (the first at once), keeping it for tiles still resident; a job that
/// failed fails its tile's water, at `frame`.
fn take_waters(state: &mut TerrainState, slots: &mut JobBudget<'_>, work: &Work, frame: u64) {
    let asked = state
        .settled
        .waters
        .keys_where(|_, watering| watering.job.is_some());
    let mut taken = false;
    for key in asked {
        if taken && work.expired() {
            break;
        }
        let Some(id) = state.settled.waters.get(&key).and_then(|watering| watering.job) else {
            continue;
        };
        let Some(output) = slots.jobs.take(id) else {
            continue;
        };
        taken = true;
        match job_output(
            SteadiedJob::KIND.name(),
            output,
            engine_terrain::jobs::steadied_output,
        ) {
            Ok(water) => {
                let resident = state.store.contains(&key);
                if let Some(watering) = state.settled.waters.get_mut(&key) {
                    watering.job = None;
                    watering.made = resident.then(|| Arc::new(water));
                }
            }
            Err(error) => {
                state.settled.waters.remove(&key);
                state.settled.waters_failed.insert(key, frame);
                state.failures.record(frame, error);
            }
        }
    }
}

/// The tiles whose water `state`'s settled water makes, in key order: the
/// resident tiles of `near` and the resident coarser tiles holding water of
/// their own, but those whose rivers something standing in them holds
/// back; while its session cooks, also every tile of `watered` and every
/// coarser tile over one of them or beside one, resident or not.
fn watering(state: &TerrainState, (watered, near): (&[TileKey], &[TileKey])) -> Vec<TileKey> {
    let store = &state.store;
    let mut tiles = store.keys_where(|key| {
        let held_back = store.watered(key).is_some_and(Option::is_some);
        !held_back
            && if key.level == 0 {
                near.binary_search(key).is_ok()
            } else {
                store.get(key).is_some_and(|tile| tile.water().is_some())
            }
    });
    if state.settled.cooking {
        let pyramid = state.terrain.pyramid;
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

/// What tile `key`'s water reads of `settled` (its tile's record being
/// `tile`), over `pyramid`: a level-0 tile's, the records of itself and the
/// tiles of `watered` around it, shared whole, its own here; a coarser
/// tile's, the summaries of those under it and around it. Where one is
/// still to come it waits; where none is here it reads nothing. Those of
/// `near` (every one while its session cooks) are made where no host holds
/// them, where a renderer `settles` water.
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
                if (dx, dz) == (0, 0) || x < 0 || z < 0 || x >= columns as i32 || z >= rows as i32
                {
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
        // The level-0 tiles under it and a tile around them.
        let up = u32::from(key.level);
        let span = |at: u16, count: u32| {
            let first = (u32::from(at) << up).saturating_sub(1);
            let last = ((u32::from(at) + 1) << up).min(count - 1);
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

/// The key of the record of a tile's water reading `reads`, on the
/// lattice of cells `cell` meters across: the keys of what it reads, never
/// their bytes.
fn water_record(reads: &Reads, cell: f32) -> RecordKey {
    let mut identity = Vec::with_capacity(36 + reads.settled.len() * 37);
    identity.extend_from_slice(&reads.tile.0);
    identity.extend_from_slice(&cell.to_le_bytes());
    for (key, record) in &reads.settled {
        identity.push(key.level);
        identity.extend_from_slice(&key.x.to_le_bytes());
        identity.extend_from_slice(&key.z.to_le_bytes());
        identity.extend_from_slice(&record.0);
    }
    RecordKey::new(
        &engine_terrain::sources::WATER,
        SteadiedJob::KIND.name(),
        &identity,
    )
}

/// Asks, as `slots` admits them, for the water of each tile `state`'s
/// settled water gives water ([`watering`]) whose reads over the recipe
/// shared as `recipe` are all here and differ from those asked for or made
/// ([`wants`]), nearest `eye` (terrain-local) first: a tile whose reads
/// changed no longer waits for what it asked for before. The water of
/// tiles it gives water no more stops, and that of tiles whose jobs failed
/// waits a while from the frame they did, `frame` now. Counts the tiles'
/// water its cooking session has left.
fn ask_waters(
    state: &mut TerrainState,
    slots: &mut JobBudget<'_>,
    (recipe, watered, near): (&str, &[TileKey], &[TileKey]),
    (eye, frame, settles): (Vec2, u64, bool),
) {
    let pyramid = state.terrain.pyramid;
    let tiles = watering(state, (watered, near));
    let settled = &mut state.settled;
    settled.waters.retain(|key, watering| {
        let kept = tiles.binary_search(key).is_ok();
        if !kept && let Some(id) = watering.job {
            slots.jobs.cancel(id);
            slots.release();
        }
        kept
    });
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
    let mut left = 0;
    for (place, key) in wanted {
        let (job, tile) = state.settled_tile(recipe, key);
        let reads = match wants(
            &state.settled,
            (&pyramid, key, tile),
            (watered, near),
            settles,
        ) {
            Wants::Here(reads) => reads,
            Wants::Coming => {
                left += 1;
                continue;
            }
            Wants::Nothing => continue,
        };
        let settled = &mut state.settled;
        if let Some(watering) = settled.waters.get(&key)
            && watering.reads == reads
        {
            left += u64::from(watering.job.is_some());
            continue;
        }
        // What it read before is not what it reads now.
        if let Some(Watering { job: Some(id), .. }) = settled.waters.remove(&key) {
            slots.jobs.cancel(id);
            slots.release();
        }
        left += 1;
        let priority = job_priority(VIEW_JOBS, place);
        if !slots.take(priority) {
            settled.deferred = true;
            continue;
        }
        let record = water_record(&reads, cell);
        let input = SteadiedJob {
            tile: job,
            cell,
            settled: reads
                .settled
                .iter()
                .map(|&(at, record)| (at, record.0))
                .collect(),
        }
        .input();
        let id = slots
            .jobs
            .submit_record(SteadiedJob::KIND.name(), input, priority, record);
        settled.waters.insert(
            key,
            Watering {
                reads,
                record,
                job: Some(id),
                made: None,
            },
        );
    }
    state.settled.waters_left = left;
}

/// Swaps the water `state`'s tiles' jobs made into the resident tiles not
/// holding it, in key order, the first at once and the rest within
/// `work`'s time and uploads: each tile current, its rivers running as its
/// bake's, and the water made over its record. The tiles given water that
/// are neither resident nor kept in memory are forgotten.
fn give_waters(state: &mut TerrainState, work: &Work) {
    let store = &state.store;
    state
        .settled
        .given
        .retain(|key, _| store.contains(key) || store.in_memory(key));
    let settled = &state.settled;
    let giving: Vec<(TileKey, Arc<Tile>, Arc<Option<TileWater>>, RecordKey)> = settled
        .waters
        .iter()
        .filter_map(|(key, watering)| {
            let water = watering.made.clone()?;
            let tile = store.get(&key)?;
            let held_back = store.watered(&key).is_some_and(Option::is_some);
            let over = settled.tiles.get(&key).map(|(_, record)| *record) == Some(watering.reads.tile);
            let holds = settled
                .given
                .get(&key)
                .is_some_and(|(given, record)| Arc::ptr_eq(given, tile) && *record == watering.record);
            (!store.loads(&key) && !held_back && over && !holds)
                .then(|| (key, Arc::clone(tile), water, watering.record))
        })
        .collect();
    let mut gave = false;
    for (key, tile, water, record) in giving {
        if gave && (work.expired() || !work.uploads()) {
            break;
        }
        gave = true;
        work.upload(2 * TILE_ARRAY_BYTES);
        let given = Arc::new(tile.holding(Option::clone(&water)));
        state.store.rewater(&key, Some(given.clone()), None);
        state.settled.given.insert(key, (given, record));
    }
}
