//! Worker jobs: self-contained work, such as a terrain tile,
//! a walk or a stack's bakes, that hosts run on workers beside the engine's
//! thread. An owner submits a job's kind and input bytes and takes its
//! output once a worker hands it back; what many jobs read (a terrain's
//! layers) is shared with every worker once, by key, before the jobs that
//! read it. Hosts hand requests to workers in the order owners make them
//! and say how many workers run them; without any (native builds, tools,
//! hosts that start none, `0` after a failure) owners do the work in place,
//! as before. Results never depend on where a job ran.
//!
//! Jobs carry a priority: workers run a job a slice at a
//! time and, between slices, the most urgent they hold; a job its owner
//! cancels stops at its next slice, so its worker takes up wanted work at
//! once. A job may hand back a part of its output before the rest,
//! as a tile its shape, for its owner to use while it runs.
//!
//! Hosts may keep outputs: a job submitted with an identity,
//! what its output depends on, takes its output from a host's cache when
//! the same build ran the same identity before, in this session or an
//! earlier one. Owners name what they share by its content
//! ([`content_hash`]), so inputs reading shares are the same bytes in every
//! session.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::prelude::*;
use serde::Serialize;

/// Ids handed out so far, across engines: a page that replaces its engine
/// never mistakes an earlier engine's output for a job of its own.
static SUBMITTED: AtomicU64 = AtomicU64::new(0);

/// A submitted job, unique across the engines of a process.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(pub u64);

/// A job to hand a worker: its id, its kind (which work workers run on
/// it), its input, its priority, lower first ([`job_priority`]), and, where
/// hosts may keep its output, its identity: the same kind
/// and identity give the same output from the same build.
#[derive(Clone, Debug, PartialEq)]
pub struct JobRequest {
    pub id: JobId,
    pub kind: &'static str,
    pub input: Vec<u8>,
    pub priority: u32,
    pub identity: Option<Vec<u8>>,
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
    Set { key: String, bytes: Vec<u8> },
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
    /// Jobs failed.
    pub failed: u64,
    /// Jobs cancelled while running.
    pub cancelled: u64,
    /// Parts handed back early.
    pub parts: u64,
    /// Keys shared.
    pub shares: u32,
    /// Bytes of the keys shared.
    pub shared_bytes: u64,
    /// Outputs the host's cache handed back instead of running their jobs.
    pub cache_hits: u64,
    /// Jobs the host's cache lacked.
    pub cache_misses: u64,
    /// Bytes the host's cache holds.
    pub cache_bytes: u64,
    /// Verifying, cached outputs differing from a fresh run.
    pub cache_mismatches: u64,
}

/// Jobs handed to the host's workers. Inserted by default,
/// with no workers until a host says how many it runs.
#[derive(Resource)]
pub struct Jobs {
    workers: u32,
    requests: Vec<JobRequest>,
    shares: Vec<JobShare>,
    // Jobs handed out and not done, and those handed out that their owners
    // cancelled, for hosts to stop.
    running: HashSet<JobId>,
    cancels: Vec<JobId>,
    done: HashMap<JobId, Result<Vec<u8>, String>>,
    // Parts of running jobs' outputs handed back early,
    // the latest of each.
    parts: HashMap<JobId, Vec<u8>>,
    // The host's cache counters: hits, misses, bytes and
    // mismatches.
    cache: [u64; 4],
    // What is shared now, by key: its size.
    shared: HashMap<String, u64>,
    stats: JobStats,
}

impl Default for Jobs {
    fn default() -> Self {
        Self {
            workers: 0,
            requests: Vec::new(),
            shares: vec![JobShare::Clear],
            running: HashSet::new(),
            cancels: Vec::new(),
            done: HashMap::new(),
            parts: HashMap::new(),
            cache: [0; 4],
            shared: HashMap::new(),
            stats: JobStats::default(),
        }
    }
}

impl Jobs {
    /// Workers jobs run on; with none, owners do the work themselves.
    pub fn workers(&self) -> u32 {
        self.workers
    }

    /// Jobs submitted and not yet done.
    pub fn running(&self) -> usize {
        self.running.len()
    }

    /// Hosts: how many workers run jobs. Going to none fails every job
    /// handed out, so owners take the work back.
    pub fn set_workers(&mut self, workers: u32) {
        self.workers = workers;
        if workers == 0 {
            self.requests.clear();
            self.cancels.clear();
            for id in std::mem::take(&mut self.running) {
                self.finish(id, Err("no workers".into()));
            }
        }
    }

    /// Submits a job of `kind` on `input` at `priority` ([`job_priority`]);
    /// [`Self::take`] its output once done. Without workers it fails at
    /// once.
    pub fn submit(&mut self, kind: &'static str, input: Vec<u8>, priority: u32) -> JobId {
        self.submit_with(kind, input, priority, None)
    }

    /// [`Self::submit`] for a job whose output hosts may keep and hand back
    /// for the same `identity` without running it: what its
    /// output depends on, as the same build computes it.
    pub fn submit_cacheable(
        &mut self,
        kind: &'static str,
        input: Vec<u8>,
        priority: u32,
        identity: Vec<u8>,
    ) -> JobId {
        self.submit_with(kind, input, priority, Some(identity))
    }

    fn submit_with(
        &mut self,
        kind: &'static str,
        input: Vec<u8>,
        priority: u32,
        identity: Option<Vec<u8>>,
    ) -> JobId {
        let id = JobId(SUBMITTED.fetch_add(1, Ordering::Relaxed) + 1);
        self.stats.submitted += 1;
        if self.workers == 0 {
            self.finish(id, Err("no workers".into()));
            return id;
        }
        self.running.insert(id);
        self.requests.push(JobRequest {
            id,
            kind,
            input,
            priority,
            identity,
        });
        id
    }

    /// Job `id`'s output once done, taken: `Err` when it failed.
    pub fn take(&mut self, id: JobId) -> Option<Result<Vec<u8>, String>> {
        self.done.remove(&id)
    }

    /// A part of running job `id`'s output handed back before the rest,
    /// taken; parts not taken before the job is done are
    /// dropped.
    pub fn take_part(&mut self, id: JobId) -> Option<Vec<u8>> {
        self.parts.remove(&id)
    }

    /// Drops job `id`: a worker running it stops at its next slice,
    /// and its output, if any, is thrown away.
    pub fn cancel(&mut self, id: JobId) {
        self.parts.remove(&id);
        if self.done.remove(&id).is_some() || !self.running.remove(&id) {
            return;
        }
        self.stats.cancelled += 1;
        match self.requests.iter().position(|request| request.id == id) {
            Some(index) => {
                self.requests.remove(index);
            }
            None => self.cancels.push(id),
        }
    }

    /// Shares `bytes` with every worker as `key`, replacing what it held:
    /// jobs submitted after read them.
    pub fn share(&mut self, key: String, bytes: Vec<u8>) {
        self.shared.insert(key.clone(), bytes.len() as u64);
        self.shares.push(JobShare::Set { key, bytes });
    }

    /// Whether `key` is shared now.
    pub fn is_shared(&self, key: &str) -> bool {
        self.shared.contains_key(key)
    }

    /// Drops what is shared as `key` from every worker.
    pub fn unshare(&mut self, key: String) {
        if self.shared.remove(&key).is_some() {
            self.shares.push(JobShare::Remove { key });
        }
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

    /// Hosts: moves the jobs handed out that their owners cancelled into
    /// `out`: their workers should stop them.
    pub fn take_cancels(&mut self, out: &mut Vec<JobId>) {
        out.append(&mut self.cancels);
    }

    /// Hosts: job `id`'s output from its worker, or why it failed; ignored
    /// for jobs cancelled since.
    pub fn complete(&mut self, id: JobId, output: Result<Vec<u8>, String>) {
        if self.running.remove(&id) {
            self.finish(id, output);
        }
    }

    /// Hosts: a part of running job `id`'s output its worker handed back
    /// before the rest, replacing an earlier one not taken;
    /// ignored for jobs cancelled or done since.
    pub fn part(&mut self, id: JobId, part: Vec<u8>) {
        if self.running.contains(&id) {
            self.stats.parts += 1;
            self.parts.insert(id, part);
        }
    }

    /// Hosts: their output cache's counters: outputs handed
    /// back from it, jobs it lacked, its bytes and, verifying, outputs that
    /// differed from a fresh run.
    pub fn report_cache(&mut self, hits: u64, misses: u64, bytes: u64, mismatches: u64) {
        self.cache = [hits, misses, bytes, mismatches];
    }

    fn finish(&mut self, id: JobId, output: Result<Vec<u8>, String>) {
        self.parts.remove(&id);
        match &output {
            Ok(_) => self.stats.completed += 1,
            Err(_) => self.stats.failed += 1,
        }
        self.done.insert(id, output);
    }

    pub fn stats(&self) -> JobStats {
        JobStats {
            workers: self.workers,
            running: self.running.len() as u32,
            shares: self.shared.len() as u32,
            shared_bytes: self.shared.values().sum(),
            cache_hits: self.cache[0],
            cache_misses: self.cache[1],
            cache_bytes: self.cache[2],
            cache_mismatches: self.cache[3],
            ..self.stats
        }
    }
}
