use crate::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::prelude::*;
use engine_terrain::jobs::TerrainWorker;
use serde::Serialize;

static SUBMITTED: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RecordKey(pub [u8; 32]);

impl RecordKey {
    pub fn new(source: &[u8; 32], kind: &str, identity: &[u8]) -> Self {
        Self(engine_sources::record_key(source, kind, identity))
    }

    pub fn hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

pub const MISSING_RECORD: &str = "no store holds the record";

#[derive(Clone, Copy, Debug)]
pub enum RecordParity {
    Exact,
    Within(fn(&[u8], &[u8]) -> bool),
}

impl RecordParity {
    pub fn holds(self, handed: &[u8], made: &[u8]) -> bool {
        match self {
            Self::Exact => handed == made,
            Self::Within(alike) => alike(handed, made),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct JobRequest {
    pub id: JobId,
    pub kind: &'static str,
    pub input: Vec<u8>,
    pub priority: u32,
    pub record: Option<RecordKey>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordFetch {
    pub id: JobId,
    pub record: RecordKey,
    pub priority: u32,
}

pub const URGENT_JOBS: u8 = 0;
pub const GAMEPLAY_JOBS: u8 = 1;
pub const VIEW_JOBS: u8 = 2;
pub const IDLE_JOBS: u8 = 3;

pub const fn job_priority(class: u8, place: u32) -> u32 {
    ((class as u32) << 24) | if place > 0xff_ffff { 0xff_ffff } else { place }
}

pub const fn job_class(priority: u32) -> u8 {
    (priority >> 24) as u8
}

pub fn content_hash(bytes: &[u8]) -> u128 {
    const C1: u64 = 0x87c3_7b91_1142_53d5;
    const C2: u64 = 0x4cf5_ad43_2745_937f;
    let mix = |mut k: u64| {
        k ^= k >> 33;
        k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
        k ^= k >> 33;
        k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        k ^ (k >> 33)
    };
    let word = |bytes: &[u8]| {
        let mut word = [0_u8; 8];
        word[..bytes.len()].copy_from_slice(bytes);
        u64::from_le_bytes(word)
    };
    let (mut h1, mut h2) = (0_u64, 0_u64);
    let (blocks, tail) = bytes.as_chunks::<16>();
    for block in blocks {
        let (k1, k2) = (word(&block[..8]), word(&block[8..]));
        h1 ^= k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
        h1 = h1
            .rotate_left(27)
            .wrapping_add(h2)
            .wrapping_mul(5)
            .wrapping_add(0x52dc_e729);
        h2 ^= k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
        h2 = h2
            .rotate_left(31)
            .wrapping_add(h1)
            .wrapping_mul(5)
            .wrapping_add(0x3849_5ab5);
    }
    if tail.len() > 8 {
        h2 ^= word(&tail[8..])
            .wrapping_mul(C2)
            .rotate_left(33)
            .wrapping_mul(C1);
    }
    if !tail.is_empty() {
        h1 ^= word(&tail[..tail.len().min(8)])
            .wrapping_mul(C1)
            .rotate_left(31)
            .wrapping_mul(C2);
    }
    let len = bytes.len() as u64;
    h1 ^= len;
    h2 ^= len;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = mix(h1);
    h2 = mix(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    (u128::from(h2) << 64) | u128::from(h1)
}

#[derive(Clone, Debug, PartialEq)]
pub enum JobShare {
    Clear,
    Set { key: String, bytes: Arc<[u8]> },
    Remove { key: String },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct JobStats {
    pub workers: u32,
    pub running: u32,
    pub submitted: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub shares: u32,
    pub shared_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_bytes: u64,
    pub cache_mismatches: u64,
    pub held: u32,
    pub in_place: u64,
}

#[derive(Resource)]
pub struct Jobs {
    workers: u32,
    hosted: bool,
    verifying: bool,
    requests: Vec<JobRequest>,
    fetches: Vec<RecordFetch>,
    keeps: Vec<(RecordKey, Vec<u8>, RecordParity)>,
    shares: Vec<JobShare>,
    running: HashSet<JobId>,
    cancels: Vec<JobId>,
    done: HashMap<JobId, Result<Vec<u8>, String>>,
    cache: [u64; 4],
    held: u32,
    shared: Vec<(String, Arc<[u8]>)>,
    sharing: u64,
    here: Option<(TerrainWorker, Vec<String>, u64)>,
    stats: JobStats,
    last_failure: Option<String>,
}

impl Default for Jobs {
    fn default() -> Self {
        Self {
            workers: 0,
            hosted: false,
            verifying: false,
            requests: Vec::new(),
            fetches: Vec::new(),
            keeps: Vec::new(),
            shares: vec![JobShare::Clear],
            running: HashSet::default(),
            cancels: Vec::new(),
            done: HashMap::default(),
            cache: [0; 4],
            held: 0,
            shared: Vec::new(),
            sharing: 0,
            here: None,
            stats: JobStats::default(),
            last_failure: None,
        }
    }
}

impl Jobs {
    pub fn workers(&self) -> u32 {
        self.workers
    }

    pub fn running(&self) -> usize {
        self.running.len()
    }

    pub fn verifying(&self) -> bool {
        self.verifying
    }

    pub fn set_verifying(&mut self, verifying: bool) {
        self.verifying = verifying;
    }

    pub fn hosted(&self) -> bool {
        self.hosted
    }

    pub fn set_workers(&mut self, workers: u32) {
        self.workers = workers;
        self.hosted = true;
    }

    pub fn submit(&mut self, kind: &'static str, input: Vec<u8>, priority: u32) -> JobId {
        self.submit_with(kind, input, priority, None)
    }

    pub fn submit_record(
        &mut self,
        kind: &'static str,
        input: Vec<u8>,
        priority: u32,
        record: RecordKey,
    ) -> JobId {
        self.submit_with(kind, input, priority, Some(record))
    }

    fn submit_with(
        &mut self,
        kind: &'static str,
        input: Vec<u8>,
        priority: u32,
        record: Option<RecordKey>,
    ) -> JobId {
        let id = next_id();
        self.stats.submitted += 1;
        if !self.hosted {
            let output = self.run_here(kind, &input);
            self.finish(id, output);
            return id;
        }
        self.running.insert(id);
        self.requests.push(JobRequest {
            id,
            kind,
            input,
            priority,
            record,
        });
        id
    }

    pub fn fetch(&mut self, record: RecordKey, priority: u32) -> JobId {
        let id = next_id();
        self.stats.submitted += 1;
        if !self.hosted {
            self.finish(id, Err(MISSING_RECORD.to_owned()));
            return id;
        }
        self.running.insert(id);
        self.fetches.push(RecordFetch {
            id,
            record,
            priority,
        });
        id
    }

    pub fn keep(&mut self, record: RecordKey, bytes: Vec<u8>) {
        self.keep_with(record, bytes, RecordParity::Exact);
    }

    pub fn keep_with(&mut self, record: RecordKey, bytes: Vec<u8>, parity: RecordParity) {
        if self.hosted {
            self.keeps.push((record, bytes, parity));
        }
    }

    pub fn take(&mut self, id: JobId) -> Option<Result<Vec<u8>, String>> {
        self.done.remove(&id)
    }

    pub fn cancel(&mut self, id: JobId) {
        if self.done.remove(&id).is_some() || !self.running.remove(&id) {
            return;
        }
        self.stats.cancelled += 1;
        if let Some(index) = self.requests.iter().position(|request| request.id == id) {
            self.requests.remove(index);
        } else if let Some(index) = self.fetches.iter().position(|fetch| fetch.id == id) {
            self.fetches.remove(index);
        } else {
            self.cancels.push(id);
        }
    }

    pub fn share(&mut self, key: String, bytes: impl Into<Arc<[u8]>>) {
        let bytes: Arc<[u8]> = bytes.into();
        self.shared.retain(|(shared, _)| *shared != key);
        self.shared.push((key.clone(), bytes.clone()));
        self.sharing += 1;
        self.shares.push(JobShare::Set { key, bytes });
    }

    pub fn is_shared(&self, key: &str) -> bool {
        self.shared.iter().any(|(shared, _)| shared == key)
    }

    pub fn unshare(&mut self, key: String) {
        let before = self.shared.len();
        self.shared.retain(|(shared, _)| *shared != key);
        if self.shared.len() < before {
            self.sharing += 1;
            self.shares.push(JobShare::Remove { key });
        }
    }

    pub fn run_here(&mut self, kind: &str, input: &[u8]) -> Result<Vec<u8>, String> {
        self.stats.in_place += 1;
        if engine_tables::TableWorker::runs(kind) || engine_textures::TextureWorker::runs(kind) {
            let _profile = engine_profile::span("cpu", "job_in_place");
            if let Some(output) =
                engine_tables::run(kind, input).or_else(|| engine_textures::run(kind, input))
            {
                return output;
            }
        }
        let (shared, sharing) = (&self.shared, self.sharing);
        let (worker, held, synced) = self
            .here
            .get_or_insert_with(|| (TerrainWorker::new(), Vec::new(), 0));
        if *synced != sharing {
            let current: HashSet<&str> = shared.iter().map(|(key, _)| key.as_str()).collect();
            for key in held.iter().filter(|key| !current.contains(key.as_str())) {
                worker.unshare(key);
            }
            held.retain(|key| current.contains(key.as_str()));
            let holding: HashSet<String> = held.iter().cloned().collect();
            for (key, bytes) in shared {
                if !holding.contains(key) {
                    worker.share(key, Arc::clone(bytes))?;
                    held.push(key.clone());
                }
            }
            *synced = sharing;
        }
        let _profile = engine_profile::span("cpu", "job_in_place");
        worker.run(kind, input)
    }

    pub fn take_shares(&mut self, out: &mut Vec<JobShare>) {
        out.append(&mut self.shares);
    }

    pub fn take_requests(&mut self, out: &mut Vec<JobRequest>) {
        out.append(&mut self.requests);
    }

    pub fn take_fetches(&mut self, out: &mut Vec<RecordFetch>) {
        out.append(&mut self.fetches);
    }

    pub fn take_keeps(&mut self, out: &mut Vec<(RecordKey, Vec<u8>, RecordParity)>) {
        out.append(&mut self.keeps);
    }

    pub fn take_cancels(&mut self, out: &mut Vec<JobId>) {
        out.append(&mut self.cancels);
    }

    pub fn complete(&mut self, id: JobId, output: Result<Vec<u8>, String>) {
        if self.running.remove(&id) {
            self.finish(id, output);
        }
    }

    pub fn report_cache(&mut self, hits: u64, misses: u64, bytes: u64, mismatches: u64) {
        self.cache = [hits, misses, bytes, mismatches];
    }

    pub fn report_held(&mut self, held: u32) {
        self.held = held;
    }

    fn finish(&mut self, id: JobId, output: Result<Vec<u8>, String>) {
        match &output {
            Ok(_) => self.stats.completed += 1,
            Err(message) if message == MISSING_RECORD => {}
            Err(message) => {
                self.stats.failed += 1;
                self.last_failure = Some(message.clone());
            }
        }
        self.done.insert(id, output);
    }

    pub fn last_failure(&self) -> Option<&str> {
        self.last_failure.as_deref()
    }

    pub fn stats(&self) -> JobStats {
        JobStats {
            workers: self.workers,
            running: self.running.len() as u32,
            shares: self.shared.len() as u32,
            shared_bytes: self
                .shared
                .iter()
                .map(|(_, bytes)| bytes.len() as u64)
                .sum(),
            cache_hits: self.cache[0],
            cache_misses: self.cache[1],
            cache_bytes: self.cache[2],
            cache_mismatches: self.cache[3],
            held: self.held,
            ..self.stats
        }
    }
}

#[derive(Resource, Clone, Debug, Default)]
pub struct Cooking {
    left: Vec<(&'static str, u64)>,
}

impl Cooking {
    pub fn left(&mut self, feature: &'static str, left: u64) {
        match self.left.iter_mut().find(|(name, _)| *name == feature) {
            Some((_, count)) => *count = left,
            None => self.left.push((feature, left)),
        }
    }

    pub fn remaining(&self) -> u64 {
        self.left.iter().map(|(_, left)| left).sum()
    }

    pub fn features(&self) -> &[(&'static str, u64)] {
        &self.left
    }
}

fn next_id() -> JobId {
    JobId(SUBMITTED.fetch_add(1, Ordering::Relaxed) + 1)
}

pub(crate) fn job_output<T, E: std::fmt::Display>(
    kind: &str,
    output: Result<Vec<u8>, String>,
    decode: impl FnOnce(&[u8]) -> Result<T, E>,
) -> Result<T, String> {
    let bytes = output.map_err(|error| format!("{kind}: {error}"))?;
    decode(&bytes).map_err(|error| format!("{kind}: its output does not decode: {error}"))
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct JobFailures {
    pub(crate) count: u64,
    pub(crate) last: Option<(u64, String)>,
}

impl JobFailures {
    pub(crate) fn record(&mut self, frame: u64, error: String) {
        self.count += 1;
        self.last = Some((frame, error));
    }
}

pub(crate) struct JobBudget<'a> {
    pub(crate) jobs: &'a mut Jobs,
    slots: usize,
    urgent: usize,
}

impl JobBudget<'_> {
    pub(crate) fn of(jobs: &mut Jobs) -> JobBudget<'_> {
        let workers = jobs.workers().max(1) as usize;
        let running = jobs.running();
        JobBudget {
            slots: (workers * 2).saturating_sub(running),
            urgent: (workers * 3).saturating_sub(running),
            jobs,
        }
    }

    pub(crate) fn take(&mut self, priority: u32) -> bool {
        let free = if job_class(priority) == URGENT_JOBS {
            self.urgent
        } else {
            self.slots
        };
        if free == 0 {
            return false;
        }
        self.slots = self.slots.saturating_sub(1);
        self.urgent = self.urgent.saturating_sub(1);
        true
    }

    pub(crate) fn release(&mut self) {
        self.slots += 1;
        self.urgent += 1;
    }
}
