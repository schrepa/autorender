use super::*;
use crate::TileShading;
use crate::jobs::{ShadingJob, ShadingReads, shading_key, shading_output, share_shading};
use engine_core::MISSING_RECORD;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Fetching(JobId),
    Making(JobId),
}

impl Phase {
    fn job(self) -> JobId {
        match self {
            Self::Fetching(id) | Self::Making(id) => id,
        }
    }
}

#[derive(Clone, Copy)]
struct Asked {
    phase: Phase,
    record: RecordKey,
}

#[derive(Clone, Copy)]
struct Want {
    key: TileKey,
    priority: u32,
    held: bool,
}

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
    pub(super) fn asked(&self) -> Vec<JobId> {
        self.asked
            .iter()
            .map(|(_, asked)| asked.phase.job())
            .collect()
    }

    pub(super) fn counts(&self) -> (u32, u32) {
        (self.shaded, self.asked.len() as u32)
    }

    pub(super) fn left(&self) -> u64 {
        self.left
    }

    pub(super) fn shares(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.sharing.iter().cloned().collect();
        keys.sort_unstable();
        keys
    }

    pub(super) fn holds_share(&self, key: &str) -> bool {
        self.sharing.contains(key)
    }

    fn share(&mut self, jobs: &mut Jobs, key: String, bytes: impl FnOnce() -> Vec<u8>) {
        if !self.sharing.insert(key.clone()) {
            return;
        }
        if !jobs.is_shared(&key) {
            jobs.share(key, bytes());
        }
    }

    fn has(&self, key: TileKey, record: RecordKey, held: bool) -> bool {
        if held {
            self.held.get(&key).is_some_and(|(_, at)| *at == record)
        } else {
            self.arrived.get(&key) == Some(&record)
        }
    }

    fn made(&self, key: TileKey, record: RecordKey) -> bool {
        key.level == 1
            || self.missing.contains(&record)
            || (self.cooking && self.arrived.get(&key) != Some(&record))
    }
}

impl TerrainState {
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
                    .map(|at| (at, self.shading_record(recipe, at).hash))
                    .collect(),
            )
        };
        let job = ShadingJob {
            recipe: recipe.to_owned(),
            key,
            reads,
        };
        let record = RecordKey::new(&crate::SOURCE, ShadingJob::KIND.name(), &job.identity());
        self.shading.found.insert(key, (job, record));
        record
    }
}

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
            Err(error) if matches!(asked.phase, Phase::Fetching(_)) && error == MISSING_RECORD => {
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
                state.failures.record(
                    frame,
                    format!("{kind}: its output does not decode: {error}"),
                );
            }
        }
    }
}

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
    let mut list =
        |wants: &mut Vec<Want>, key: TileKey, class: u8, held: bool| match listed.get(&key) {
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

fn keep(state: &mut TerrainState, slots: &mut JobBudget<'_>, wants: &[Want]) {
    let mut wanted: TileMap<bool> = TileMap::default();
    for want in wants {
        wanted.insert(want.key, want.held);
    }
    let store = &state.store;
    let shading = &mut state.shading;
    let found = &shading.found;
    let current =
        |key: &TileKey, record: RecordKey| found.get(key).is_some_and(|(_, now)| *now == record);
    let stale = shading
        .asked
        .keys_where(|key, asked| !wanted.contains_key(key) || !current(key, asked.record));
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
                .map(|&(at, hash)| {
                    (
                        at,
                        RecordKey {
                            hash,
                            kind: ShadingJob::KIND.name(),
                        },
                    )
                })
                .collect(),
            ShadingReads::Tiles(_) => Vec::new(),
        };
        if made && !reads.iter().all(|&(at, read)| shading.has(at, read, true)) {
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
                state.shading.share(slots.jobs, shading_key(&read.hash), || {
                    share_shading(&crate::codec::encode(held.as_ref()))
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
