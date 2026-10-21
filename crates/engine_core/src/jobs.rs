//! Worker jobs: self-contained work, such as a terrain tile, a tile's walk
//! graph portals, a stack's bakes or a table the renderer reads, that hosts
//! run on workers beside the engine's thread. An owner submits a job's kind
//! and input bytes and takes its output once a worker hands it back; what
//! many jobs read (a terrain's layers) is shared with every worker once, by
//! key, before the jobs that read it. Hosts hand requests to workers in the
//! order owners make them and say how many workers run them; a host
//! without workers runs them in place, and so does an engine no host
//! serves. Results never depend on where a job ran.
//!
//! Jobs carry a priority: workers run the most urgent they hold first, and
//! a job its owner cancels is dropped, so its worker takes up wanted work.
//!
//! Records: a job submitted with a [`RecordKey`], a hash of the code
//! making its output and of everything that output depends on, takes its
//! output from the records hosts hold where one has it (the cook's packs,
//! then what the host kept before, in this session or an earlier one), and
//! the host keeps what a worker makes. Owners name what they share by its
//! content ([`content_hash`]), so keys reading shares are the same in every
//! session. A record no job makes is fetched alone ([`Jobs::fetch`]), and
//! an owner that made one itself hands it to the host to keep
//! ([`Jobs::keep`]).

use crate::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::prelude::*;
use engine_terrain::jobs::TerrainWorker;
use serde::Serialize;

/// Ids handed out so far, across engines: a page that replaces its engine
/// never mistakes an earlier engine's output for a job of its own.
static SUBMITTED: AtomicU64 = AtomicU64::new(0);

/// A submitted job, unique across the engines of a process.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(pub u64);

/// What hosts keep a record by, in their stores and the cook's packs: a
/// hash of the code making it (its crate's `SOURCE`), its kind and its
/// identity, everything its bytes depend on. The same key names the same
/// bytes in every build of that code, on every host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RecordKey(pub [u8; 32]);

impl RecordKey {
    /// The key of the record of `kind` on `identity` that code hashing to
    /// `source` makes.
    pub fn new(source: &[u8; 32], kind: &str, identity: &[u8]) -> Self {
        Self(engine_sources::record_key(source, kind, identity))
    }

    /// Its 64 lowercase hexadecimal digits, which stores name its file by.
    pub fn hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

/// Why a record fetched alone came back without bytes: no store or pack
/// holds it.
pub const MISSING_RECORD: &str = "no store holds the record";

/// A job to hand a worker: its id, its kind (which work workers run on
/// it), its input, its priority, lower first ([`job_priority`]), and, where
/// its output is a record, the record's key: the same key gives the same
/// output.
#[derive(Clone, Debug, PartialEq)]
pub struct JobRequest {
    pub id: JobId,
    pub kind: &'static str,
    pub input: Vec<u8>,
    pub priority: u32,
    pub record: Option<RecordKey>,
}

/// A record fetched alone, no job making it: its request's id, the
/// record's key and the request's priority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordFetch {
    pub id: JobId,
    pub record: RecordKey,
    pub priority: u32,
}

/// Classes of jobs, most urgent first: what a view or its
/// gameplay cannot do without (bakes, the coarsest tiles, the tiles a view
/// draws and shows, those under focus entities); gameplay's searches; and
/// the rest of what views load.
pub const URGENT_JOBS: u8 = 0;
pub const GAMEPLAY_JOBS: u8 = 1;
pub const VIEW_JOBS: u8 = 2;
/// Work ahead of need while nothing else waits.
pub const IDLE_JOBS: u8 = 3;

/// A job's priority: its `class`, then its `place` among the class's jobs
/// (at most 2^24 - 1).
pub const fn job_priority(class: u8, place: u32) -> u32 {
    ((class as u32) << 24) | if place > 0xff_ffff { 0xff_ffff } else { place }
}

/// A job priority's class.
pub const fn job_class(priority: u32) -> u8 {
    (priority >> 24) as u8
}

/// A 128-bit hash of `bytes` (MurmurHash3, x64 128-bit, seed 0) naming
/// content by what it holds: shares named so make the jobs
/// reading them the same bytes in every session, which hosts' caches key
/// outputs by.
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

/// What every worker should hold, in order with the requests: nothing an
/// earlier engine shared (an engine's first share), bytes shared as a key,
/// or a key dropped.
#[derive(Clone, Debug, PartialEq)]
pub enum JobShare {
    Clear,
    Set { key: String, bytes: Arc<[u8]> },
    Remove { key: String },
}

/// Workers, their jobs and the host's output cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct JobStats {
    /// Workers the host runs.
    pub workers: u32,
    /// Jobs handed out and not done.
    pub running: u32,
    /// Jobs submitted.
    pub submitted: u64,
    /// Jobs completed.
    pub completed: u64,
    /// Jobs failed. A record fetched alone that no pack or store holds
    /// ([`MISSING_RECORD`]) is no failure: its owner makes it.
    pub failed: u64,
    /// Jobs cancelled while running.
    pub cancelled: u64,
    /// Keys shared.
    pub shares: u32,
    /// Bytes of the keys shared.
    pub shared_bytes: u64,
    /// Records the host's packs and store handed back instead of running
    /// their jobs.
    pub cache_hits: u64,
    /// Records they lacked.
    pub cache_misses: u64,
    /// Bytes the host's store holds.
    pub cache_bytes: u64,
    /// Verifying, records differing from a fresh run.
    pub cache_mismatches: u64,
    /// Jobs the host's scheduler holds: on its workers, or looked up in its
    /// packs and store. None once every job handed out ended, an earlier
    /// engine's included.
    pub held: u32,
    /// Jobs run in place: by a host without workers, or an engine no host
    /// serves.
    pub in_place: u64,
}

/// Jobs handed to the host's workers. Inserted by default; until a host
/// serves it, jobs run in place as they are submitted.
#[derive(Resource)]
pub struct Jobs {
    workers: u32,
    // A host takes the requests, the fetches and the records kept.
    hosted: bool,
    // The host makes every record it hands back again and compares.
    verifying: bool,
    requests: Vec<JobRequest>,
    fetches: Vec<RecordFetch>,
    keeps: Vec<(RecordKey, Vec<u8>)>,
    shares: Vec<JobShare>,
    // Jobs handed out and not done, and those handed out that their owners
    // cancelled, for hosts to stop.
    running: HashSet<JobId>,
    cancels: Vec<JobId>,
    done: HashMap<JobId, Result<Vec<u8>, String>>,
    // The host's cache counters: hits, misses, bytes and
    // mismatches.
    cache: [u64; 4],
    // Jobs the host's scheduler holds.
    held: u32,
    // What is shared now, in the order shared: the bytes jobs run in place
    // read.
    shared: Vec<(String, Arc<[u8]>)>,
    // Where jobs run in place, once one does, and the keys it holds.
    here: Option<(TerrainWorker, Vec<String>)>,
    stats: JobStats,
    // The latest failed job's message.
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
            here: None,
            stats: JobStats::default(),
            last_failure: None,
        }
    }
}

impl Jobs {
    /// Workers jobs run on; with none, they run in place.
    pub fn workers(&self) -> u32 {
        self.workers
    }

    /// Jobs submitted and not yet done.
    pub fn running(&self) -> usize {
        self.running.len()
    }

    /// Whether the host verifies the records it hands back: jobs it runs
    /// anyway, and records fetched alone their owners make again and keep
    /// ([`Self::keep`]), which it compares.
    pub fn verifying(&self) -> bool {
        self.verifying
    }

    /// Hosts: whether they verify the records they hand back.
    pub fn set_verifying(&mut self, verifying: bool) {
        self.verifying = verifying;
    }

    /// Whether a host serves the jobs: an engine no host serves draws no
    /// frames.
    pub fn hosted(&self) -> bool {
        self.hosted
    }

    /// Hosts: how many workers run jobs, the host serving them from now
    /// on; one without workers runs them in place ([`Self::run_here`]).
    pub fn set_workers(&mut self, workers: u32) {
        self.workers = workers;
        self.hosted = true;
    }

    /// Submits a job of `kind` on `input` at `priority` ([`job_priority`]);
    /// [`Self::take`] its output once done.
    pub fn submit(&mut self, kind: &'static str, input: Vec<u8>, priority: u32) -> JobId {
        self.submit_with(kind, input, priority, None)
    }

    /// [`Self::submit`] for a job whose output is the record `record`:
    /// hosts hand back the record where they hold it, without running the
    /// job, and keep what the job makes.
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

    /// Fetches the record `record`, which no job makes, at `priority`:
    /// [`Self::take`] its bytes once a host's packs or store hand them back,
    /// or [`MISSING_RECORD`] where none holds it.
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

    /// Hands hosts `bytes`, the record `record` its owner made itself, to
    /// keep: a later fetch of it, in this session or another, takes them. A
    /// verifying host compares a record it handed back with the bytes its
    /// owner made again.
    pub fn keep(&mut self, record: RecordKey, bytes: Vec<u8>) {
        if self.hosted {
            self.keeps.push((record, bytes));
        }
    }

    /// Job `id`'s output once done, taken: `Err` when it failed.
    pub fn take(&mut self, id: JobId) -> Option<Result<Vec<u8>, String>> {
        self.done.remove(&id)
    }

    /// Drops job `id`: a worker running it stops, and its output, if any,
    /// is thrown away.
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

    /// Shares `bytes` with every worker as `key`, replacing what it held:
    /// jobs submitted after read them.
    pub fn share(&mut self, key: String, bytes: Vec<u8>) {
        let bytes: Arc<[u8]> = bytes.into();
        self.shared.retain(|(shared, _)| *shared != key);
        self.shared.push((key.clone(), bytes.clone()));
        self.shares.push(JobShare::Set { key, bytes });
    }

    /// Whether `key` is shared now.
    pub fn is_shared(&self, key: &str) -> bool {
        self.shared.iter().any(|(shared, _)| shared == key)
    }

    /// Drops what is shared as `key` from every worker.
    pub fn unshare(&mut self, key: String) {
        let before = self.shared.len();
        self.shared.retain(|(shared, _)| *shared != key);
        if self.shared.len() < before {
            self.shares.push(JobShare::Remove { key });
        }
    }

    /// Runs job `kind` on `input` here, whole, over what is shared now:
    /// what a host without workers does with the jobs no record answers,
    /// and an engine no host serves with every job. Each kind runs as
    /// workers run it: a table's by `engine_tables`, a terrain's by its
    /// worker.
    pub fn run_here(&mut self, kind: &str, input: &[u8]) -> Result<Vec<u8>, String> {
        self.stats.in_place += 1;
        if engine_tables::TableWorker::runs(kind) {
            let _profile = engine_profile::span("cpu", "job_in_place");
            if let Some(output) = engine_tables::run(kind, input) {
                return output;
            }
        }
        let shared = &self.shared;
        let (worker, held) = self
            .here
            .get_or_insert_with(|| (TerrainWorker::new(), Vec::new()));
        // It holds what is shared now, in the order shared.
        let current = |key: &String| shared.iter().any(|(shared, _)| shared == key);
        for key in held.iter().filter(|key| !current(key)) {
            worker.unshare(key);
        }
        held.retain(current);
        for (key, bytes) in shared {
            if !held.contains(key) {
                worker.share(key, bytes)?;
                held.push(key.clone());
            }
        }
        let _profile = engine_profile::span("cpu", "job_in_place");
        worker.run(kind, input)
    }

    /// Hosts: moves the shares to hand every worker into `out`, in order;
    /// hand them before [`Self::take_requests`]' requests.
    pub fn take_shares(&mut self, out: &mut Vec<JobShare>) {
        out.append(&mut self.shares);
    }

    /// Hosts: moves the jobs to hand workers into `out`, in the order owners
    /// submitted them.
    pub fn take_requests(&mut self, out: &mut Vec<JobRequest>) {
        out.append(&mut self.requests);
    }

    /// Hosts: moves the records fetched alone into `out`, in order.
    pub fn take_fetches(&mut self, out: &mut Vec<RecordFetch>) {
        out.append(&mut self.fetches);
    }

    /// Hosts: moves the records owners made themselves, to keep, into
    /// `out`.
    pub fn take_keeps(&mut self, out: &mut Vec<(RecordKey, Vec<u8>)>) {
        out.append(&mut self.keeps);
    }

    /// Hosts: moves the jobs handed out that their owners cancelled into
    /// `out`: their workers should stop them.
    pub fn take_cancels(&mut self, out: &mut Vec<JobId>) {
        out.append(&mut self.cancels);
    }

    /// Hosts: job `id`'s output, or why it failed; ignored for jobs
    /// cancelled since.
    pub fn complete(&mut self, id: JobId, output: Result<Vec<u8>, String>) {
        if self.running.remove(&id) {
            self.finish(id, output);
        }
    }

    /// Hosts: their packs' and store's counters: records handed back from
    /// them, records they lacked, the store's bytes and, verifying, records
    /// that differed from a fresh run.
    pub fn report_cache(&mut self, hits: u64, misses: u64, bytes: u64, mismatches: u64) {
        self.cache = [hits, misses, bytes, mismatches];
    }

    /// Hosts: the jobs their scheduler holds, on workers or in lookups.
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

    /// The latest failed job's message, if any job failed.
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

/// A session cooking: hosts insert it to make every record a scene's
/// content needs, not only what its views want, and write them into a
/// pack once nothing is left. While it is present each feature asks for
/// all its records and says how many it has left
/// ([`Cooking::left`]).
#[derive(Resource, Clone, Debug, Default)]
pub struct Cooking {
    left: Vec<(&'static str, u64)>,
}

impl Cooking {
    /// Feature `feature` has `left` records still to make.
    pub fn left(&mut self, feature: &'static str, left: u64) {
        match self.left.iter_mut().find(|(name, _)| *name == feature) {
            Some((_, count)) => *count = left,
            None => self.left.push((feature, left)),
        }
    }

    /// Records left, over every feature that said.
    pub fn remaining(&self) -> u64 {
        self.left.iter().map(|(_, left)| left).sum()
    }

    /// What each feature said it has left, in the order they first said.
    pub fn features(&self) -> &[(&'static str, u64)] {
        &self.left
    }
}

/// A new job's id.
fn next_id() -> JobId {
    JobId(SUBMITTED.fetch_add(1, Ordering::Relaxed) + 1)
}

/// A job of `kind`'s output decoded by `decode`, or why there is none: the
/// job failed, or its output does not decode.
pub(crate) fn job_output<T, E: std::fmt::Display>(
    kind: &str,
    output: Result<Vec<u8>, String>,
    decode: impl FnOnce(&[u8]) -> Result<T, E>,
) -> Result<T, String> {
    let bytes = output.map_err(|error| format!("{kind}: {error}"))?;
    decode(&bytes).map_err(|error| format!("{kind}: its output does not decode: {error}"))
}

/// An owner's jobs that failed: how many, and the last one's error with
/// the owner's frame it failed at. Owners read outputs with [`job_output`]
/// and record what failed here instead of dropping it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct JobFailures {
    pub(crate) count: u64,
    pub(crate) last: Option<(u64, String)>,
}

impl JobFailures {
    /// Counts a failure: `error`, at the owner's frame `frame`.
    pub(crate) fn record(&mut self, frame: u64, error: String) {
        self.count += 1;
        self.last = Some((frame, error));
    }
}

/// The jobs a frame may hand workers, the one rule every owner admits jobs
/// by: two in flight per worker, so each has its next ready, and a third
/// for urgent jobs, which workers take up first; less every job still
/// running, whoever submitted it. Jobs running in place count as one
/// worker's. An owner handing out jobs as views want them takes a budget
/// when it starts and a slot for each job.
pub(crate) struct JobBudget<'a> {
    pub(crate) jobs: &'a mut Jobs,
    slots: usize,
    urgent: usize,
}

impl JobBudget<'_> {
    /// What `jobs`' workers leave free now.
    pub(crate) fn of(jobs: &mut Jobs) -> JobBudget<'_> {
        let workers = jobs.workers().max(1) as usize;
        let running = jobs.running();
        JobBudget {
            slots: (workers * 2).saturating_sub(running),
            urgent: (workers * 3).saturating_sub(running),
            jobs,
        }
    }

    /// Takes a slot for a job of `priority`; whether one was free.
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

    /// Gives back the slot of a job cancelled while running.
    pub(crate) fn release(&mut self) {
        self.slots += 1;
        self.urgent += 1;
    }
}
