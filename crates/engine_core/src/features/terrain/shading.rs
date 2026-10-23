//! Terrains' shading ([`TileShading`]): the control values and normal maps
//! each terrain's tiles of level 1 and coarser shade with, level 0's
//! averaged, records the cook makes for every such tile and hosts keep. A
//! level-1 tile's is a job on a worker generating the level-0 tiles under it
//! and around it as their records hold them, under the edits reaching each;
//! a coarser tile's averages the records of the level finer under it and
//! around it, which are shared with workers by key while the jobs reading
//! them run. Each record is kept by its job, which names what it reads, so
//! an edit makes anew the records reading the ground it reaches, level by
//! level, and leaves every other as it was.
//!
//! Frames ask for the records of the resident tiles shading with them, the
//! pinned levels' first and then nearest the eye: fetched where a host
//! holds them, else made once what they read is here, each read fetched or
//! made in turn; while a session cooks, every such tile's, the finer levels
//! first, each coarser record made once the level finer under it is in. A
//! record that arrives attaches to its resident tile
//! ([`Tile::shaded`]), whose layers upload it and whose materials fade in;
//! until it does, the tile draws its heights and shows the next coarser
//! level's materials and normals over them.

use super::*;
use crate::MISSING_RECORD;
use engine_terrain::TileShading;
use engine_terrain::jobs::{ShadingJob, ShadingReads, shading_key, shading_output, share_shading};

/// Where a shading record asked for stands: fetched from a host's packs and
/// store, or made by its job, whose output the host keeps.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Fetching(JobId),
    Making(JobId),
}

impl Phase {
    /// Its job.
    fn job(self) -> JobId {
        match self {
            Self::Fetching(id) | Self::Making(id) => id,
        }
    }
}

/// A shading record asked for: where it stands, and its key.
#[derive(Clone, Copy)]
struct Asked {
    phase: Phase,
    record: RecordKey,
}

/// A tile whose shading a frame wants: its key, its job's priority, and
/// whether its record is held here once it arrives (for its resident tile,
/// or for a record made from it), not only had arrive.
#[derive(Clone, Copy)]
struct Want {
    key: TileKey,
    priority: u32,
    held: bool,
}

/// A terrain's shading: each tile's job and its record's key, over the
/// recipe and generation they were found for; the record each tile last
/// had arrive, and those held here, decoded, with their keys; the records
/// asked for, those that failed (at the frame they did) and those no host
/// holds; the keys it shares with workers; the resident tiles holding their
/// shading as it last attached them; and whether its session cooks, which
/// asks for every tile's, with how many of those it has left.
#[derive(Default)]
pub(super) struct Shadings {
    found: TileMap<(ShadingJob, RecordKey)>,
    found_for: Option<(String, Generation)>,
    arrived: TileMap<RecordKey>,
    held: TileMap<(Arc<TileShading>, RecordKey)>,
    asked: TileMap<Asked>,
    failed: TileMap<u64>,
    missing: HashSet<RecordKey>,
    sharing: HashSet<String>,
    shaded: u32,
    pub(super) cooking: bool,
    left: u64,
}

impl Shadings {
    /// The jobs of its records asked for.
    pub(super) fn asked(&self) -> Vec<JobId> {
        self.asked
            .iter()
            .map(|(_, asked)| asked.phase.job())
            .collect()
    }

    /// The resident tiles holding their shading, and the records asked for.
    pub(super) fn counts(&self) -> (u32, u32) {
        (self.shaded, self.asked.len() as u32)
    }

    /// The records its session cooking has left: every tile's whose record
    /// under the edits the ground has now has not arrived and did not fail.
    pub(super) fn left(&self) -> u64 {
        self.left
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

    /// Whether tile `key`'s record `record` is here as a want takes it:
    /// held here where `held`, else arrived.
    fn has(&self, key: TileKey, record: RecordKey, held: bool) -> bool {
        if held {
            self.held.get(&key).is_some_and(|(_, at)| *at == record)
        } else {
            self.arrived.get(&key) == Some(&record)
        }
    }

    /// Whether tile `key`'s record `record` is made rather than fetched: a
    /// level-1 tile's always (its job hands back what a host holds), a
    /// coarser one's where no host holds it, and while its session cooks
    /// where it has yet to arrive.
    fn made(&self, key: TileKey, record: RecordKey) -> bool {
        key.level == 1
            || self.missing.contains(&record)
            || (self.cooking && self.arrived.get(&key) != Some(&record))
    }
}

impl TerrainState {
    /// The key of tile `key`'s shading record over the recipe shared as
    /// `recipe`, its job found with it under the generation tiles are made
    /// from now: a level-1 tile's naming the level-0 tiles it reads with the
    /// edits reaching each, a coarser one's the records of the level finer
    /// it reads by their keys.
    fn shading_record(&mut self, recipe: &str, key: TileKey) -> RecordKey {
        let basis = (recipe.to_owned(), self.store.generation());
        if self.shading.found_for.as_ref() != Some(&basis) {
            self.shading.found = TileMap::default();
            self.shading.found_for = Some(basis);
        }
        if let Some((_, record)) = self.shading.found.get(&key) {
            return *record;
        }
        let read = TileShading::reads(&self.terrain.pyramid, key);
        let reads = if key.level == 1 {
            ShadingReads::Tiles(
                read.into_iter()
                    .map(|at| (at, self.tile_record(recipe, at).0.stamps))
                    .collect(),
            )
        } else {
            ShadingReads::Shadings(
                read.into_iter()
                    .map(|at| (at, self.shading_record(recipe, at).0))
                    .collect(),
            )
        };
        let job = ShadingJob {
            recipe: recipe.to_owned(),
            key,
            reads,
        };
        let record = RecordKey::new(
            &engine_terrain::sources::SHADING,
            ShadingJob::KIND.name(),
            &job.identity(),
        );
        self.shading.found.insert(key, (job, record));
        record
    }
}

/// Follows `state`'s shading for this frame: takes the records that
/// arrived; asks, while `slots` admits them, for those of its resident
/// tiles of level 1 and coarser, the `settings`' pinned levels' first, then
/// nearest `eye` (terrain-local), and for what the records made here read
/// (while its session cooks, for every such tile's), those of tiles no
/// longer wanted stopping; shares with workers what the records being made
/// read, the shares it lets go of going to `released`; and attaches the
/// records here to their resident tiles within `work`'s time and uploads.
/// Surroundings, which shade with their own samples, hold nothing.
pub(super) fn follow(
    state: &mut TerrainState,
    settings: &TerrainSettings,
    (slots, work): (&mut JobBudget<'_>, &Work),
    released: &mut Vec<String>,
    (eye, frame): (Vec2, u64),
) {
    let recipe = state.recipe.as_ref().map(|recipe| recipe.key.clone());
    let Some(recipe) = recipe.filter(|_| state.around.is_none()) else {
        for id in state.shading.asked() {
            slots.jobs.cancel(id);
            slots.release();
        }
        released.extend(state.shading.shares());
        let cooking = state.shading.cooking;
        state.shading = Shadings {
            cooking,
            ..Shadings::default()
        };
        return;
    };
    take_arrived(state, slots, frame);
    let wants = wanted(state, &recipe, settings.pinned_levels, eye);
    keep(state, slots, &wants);
    ask(state, slots, &wants, frame);
    release_shares(&mut state.shading, released);
    attach(state, work);
}

/// Takes the records of `state`'s shading asked for that arrived, each held
/// here with its key; a record no host holds is counted missing, made next
/// time it is wanted, and one that failed fails its tile, at `frame`.
fn take_arrived(state: &mut TerrainState, slots: &mut JobBudget<'_>, frame: u64) {
    let kind = ShadingJob::KIND.name();
    let asked: Vec<(TileKey, Asked)> = state
        .shading
        .asked
        .iter()
        .map(|(key, &asked)| (key, asked))
        .collect();
    for (key, asked) in asked {
        let Some(output) = slots.jobs.take(asked.phase.job()) else {
            continue;
        };
        let shading = &mut state.shading;
        shading.asked.remove(&key);
        let bytes = match output {
            Ok(bytes) => bytes,
            Err(error)
                if matches!(asked.phase, Phase::Fetching(_)) && error == MISSING_RECORD =>
            {
                shading.missing.insert(asked.record);
                continue;
            }
            Err(error) => {
                shading.failed.insert(key, frame);
                state.failures.record(frame, format!("{kind}: {error}"));
                continue;
            }
        };
        match shading_output(&bytes) {
            Ok(made) if made.key() == key => {
                shading.failed.remove(&key);
                shading.missing.remove(&asked.record);
                shading.arrived.insert(key, asked.record);
                shading.held.insert(key, (Arc::new(made), asked.record));
            }
            Ok(_) => {
                shading.failed.insert(key, frame);
            }
            Err(error) => {
                shading.failed.insert(key, frame);
                state
                    .failures
                    .record(frame, format!("{kind}: its output does not decode: {error}"));
            }
        }
    }
}

/// The tiles whose shading `state` wants over the recipe shared as `recipe`,
/// most important first, with their jobs' priorities: its resident tiles
/// shading with theirs, held, the coarsest `pinned` levels' first, then
/// nearest `eye` (terrain-local); while its session cooks, every tile of
/// level 1 and coarser, the finer levels first; then what the records made
/// here and not yet arrived read, held for their jobs to read, those a
/// resident tile's record reads after the others, while nothing else waits.
fn wanted(state: &mut TerrainState, recipe: &str, pinned: u32, eye: Vec2) -> Vec<Want> {
    let pyramid = state.terrain.pyramid;
    let levels = pyramid.levels();
    let size = pyramid.tile_size(0);
    let place = |key: TileKey| {
        let span = pyramid.tile_size(u32::from(key.level));
        let middle = pyramid.tile_origin(key) + Vec2::splat(0.5 * span);
        let pinned_level = u32::from(key.level) + pinned >= levels;
        (!pinned_level, (middle.distance(eye) / size) as u32, key)
    };
    let mut resident = state
        .store
        .keys_where(|key| TileShading::shades(key.level, false));
    resident.sort_unstable_by_key(|&key| place(key));
    let cooking = state.shading.cooking;
    let mut wants: Vec<Want> = Vec::with_capacity(resident.len());
    let mut listed: TileMap<usize> = TileMap::default();
    let mut list = |wants: &mut Vec<Want>, key: TileKey, class: u8, held: bool| {
        match listed.get(&key) {
            Some(&at) => wants[at].held |= held,
            None => {
                listed.insert(key, wants.len());
                let priority = job_priority(class, wants.len() as u32);
                wants.push(Want {
                    key,
                    priority,
                    held,
                });
            }
        }
    };
    for &key in &resident {
        list(&mut wants, key, VIEW_JOBS, true);
    }
    if cooking {
        for level in 1..levels {
            for key in pyramid.level_tiles(level) {
                list(&mut wants, key, VIEW_JOBS, false);
            }
        }
    }
    let class = if cooking { VIEW_JOBS } else { IDLE_JOBS };
    let mut at = 0;
    while let Some(&want) = wants.get(at) {
        at += 1;
        if want.key.level < 2 {
            continue;
        }
        let record = state.shading_record(recipe, want.key);
        let shading = &state.shading;
        if shading.has(want.key, record, false) || !shading.made(want.key, record) {
            continue;
        }
        let reads: Vec<TileKey> = match shading.found.get(&want.key).map(|(job, _)| &job.reads) {
            Some(ShadingReads::Shadings(reads)) => reads.iter().map(|&(read, _)| read).collect(),
            Some(ShadingReads::Tiles(_)) | None => Vec::new(),
        };
        for read in reads {
            list(&mut wants, read, class, true);
        }
    }
    for want in &wants {
        state.shading_record(recipe, want.key);
    }
    wants
}

/// Keeps what `state`'s shading still wants of `wants`: the records asked
/// for under their tiles' keys now (those of tiles no longer wanted, or
/// under other keys, stop, `slots` given theirs back), and the records held
/// for a want holding them or a tile kept in memory, under their keys now;
/// and counts what its session cooking has left.
fn keep(state: &mut TerrainState, slots: &mut JobBudget<'_>, wants: &[Want]) {
    let mut wanted: TileMap<bool> = TileMap::default();
    for want in wants {
        wanted.insert(want.key, want.held);
    }
    let store = &state.store;
    let shading = &mut state.shading;
    let found = &shading.found;
    let current = |key: &TileKey, record: RecordKey| {
        found.get(key).is_some_and(|(_, now)| *now == record)
    };
    let stale = shading.asked.keys_where(|key, asked| {
        !wanted.contains_key(key) || !current(key, asked.record)
    });
    for key in stale {
        if let Some(asked) = shading.asked.remove(&key) {
            slots.jobs.cancel(asked.phase.job());
            slots.release();
        }
    }
    shading.held.retain(|key, (_, record)| {
        let holding = wanted.get(key).copied().unwrap_or(false) || store.in_memory(key);
        holding && current(key, *record)
    });
    shading.arrived.retain(|key, _| wanted.contains_key(key));
    shading.failed.retain(|key, _| wanted.contains_key(key));
    shading.left = if shading.cooking {
        wants
            .iter()
            .filter(|want| {
                let record = found.get(&want.key).map(|(_, record)| *record);
                shading.arrived.get(&want.key).copied() != record
                    && !shading.failed.contains_key(&want.key)
            })
            .count() as u64
    } else {
        0
    };
}

/// Asks for the records of `wants` that `state`'s shading lacks, in their
/// order, while `slots` admits them: a record is fetched where a host may
/// hold it, and made where none does ([`Shadings::made`]), a coarser one
/// once every record it reads is held here, which it then shares with
/// workers first. A record that failed is asked for again a while after it
/// did (never while a session cooks), `frame` now.
fn ask(state: &mut TerrainState, slots: &mut JobBudget<'_>, wants: &[Want], frame: u64) {
    for want in wants {
        let key = want.key;
        let shading = &state.shading;
        let Some((job, record)) = shading.found.get(&key).map(|(job, record)| (job, *record))
        else {
            continue;
        };
        let waits = shading
            .failed
            .get(&key)
            .is_some_and(|&at| shading.cooking || frame < at + RETRY_AFTER);
        if shading.has(key, record, want.held) || shading.asked.contains_key(&key) || waits {
            continue;
        }
        let made = shading.made(key, record);
        let reads: Vec<(TileKey, RecordKey)> = match &job.reads {
            ShadingReads::Shadings(reads) => reads
                .iter()
                .map(|&(at, read)| (at, RecordKey(read)))
                .collect(),
            ShadingReads::Tiles(_) => Vec::new(),
        };
        if made && !reads.iter().all(|&(at, read)| shading.has(at, read, true)) {
            // While a session cooks, what reads a record that failed fails.
            if shading.cooking && reads.iter().any(|(at, _)| shading.failed.contains_key(at)) {
                state.shading.failed.insert(key, frame);
            }
            continue;
        }
        if !slots.take(want.priority) {
            break;
        }
        let input = job.identity();
        let phase = if made {
            for (at, read) in reads {
                let Some((held, _)) = state.shading.held.get(&at) else {
                    continue;
                };
                let held = Arc::clone(held);
                state.shading.share(slots.jobs, shading_key(&read.0), || {
                    share_shading(&engine_terrain::codec::encode(held.as_ref()))
                });
            }
            let kind = ShadingJob::KIND.name();
            Phase::Making(slots.jobs.submit_record(kind, input, want.priority, record))
        } else {
            Phase::Fetching(slots.jobs.fetch(record, want.priority))
        };
        state.shading.asked.insert(key, Asked { phase, record });
    }
}

/// Lets go of the shares no record `shading` makes reads any more into
/// `released`, for streaming to drop at the frame's end where no terrain
/// holds them.
fn release_shares(shading: &mut Shadings, released: &mut Vec<String>) {
    let mut reading: HashSet<String> = HashSet::default();
    for (key, asked) in shading.asked.iter() {
        let Phase::Making(_) = asked.phase else {
            continue;
        };
        if let Some((job, _)) = shading.found.get(&key)
            && let ShadingReads::Shadings(reads) = &job.reads
        {
            reading.extend(reads.iter().map(|(_, read)| shading_key(read)));
        }
    }
    for key in shading.shares() {
        if !reading.contains(&key) {
            shading.sharing.remove(&key);
            released.push(key);
        }
    }
}

/// Attaches the records held here to `state`'s resident tiles shading with
/// them that do not hold them yet, in key order, the first at once and the
/// rest within `work`'s time and uploads: each tile current, the record
/// the one its key names now. Counts the resident tiles holding their
/// shading.
fn attach(state: &mut TerrainState, work: &Work) {
    let keys = state
        .store
        .keys_where(|key| TileShading::shades(key.level, false));
    let mut gave = false;
    for key in keys {
        let shading = &state.shading;
        let (Some(tile), Some((held, record))) = (state.store.get(&key), shading.held.get(&key))
        else {
            continue;
        };
        let current = shading
            .found
            .get(&key)
            .is_some_and(|(_, now)| now == record);
        let holds = tile
            .shading()
            .is_some_and(|attached| Arc::ptr_eq(attached, held));
        if holds || !current || state.store.loads(&key) {
            continue;
        }
        if gave && (work.expired() || !work.uploads()) {
            break;
        }
        gave = true;
        work.upload(2 * TILE_ARRAY_BYTES);
        let tile = Arc::new(tile.shaded(Some(Arc::clone(held))));
        state.store.reshade(&key, tile);
    }
    let shaded = state.store.tiles().filter(|tile| tile.shading().is_some());
    state.shading.shaded = shaded.count() as u32;
}
