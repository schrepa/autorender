//! Worker jobs (decision 110): self-contained work, such as a terrain tile,
//! a walk or a stack's bakes, that hosts run on workers beside the engine's
//! thread. An owner submits a job's kind and input bytes and takes its
//! output once a worker hands it back; what many jobs read (a terrain's
//! layers) is shared with every worker once, by key, before the jobs that
//! read it. Hosts hand requests to workers in the order owners make them
//! and say how many workers run them; without any (native builds, tools,
//! hosts that start none, `0` after a failure) owners do the work in place,
//! as before. Results never depend on where a job ran.
//!
//! Jobs carry a priority (decision 112): workers run a job a slice at a
//! time and, between slices, the most urgent they hold; a job its owner
//! cancels stops at its next slice, so its worker takes up wanted work at
//! once.

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
/// it), its input and its priority, lower first ([`job_priority`]).
#[derive(Clone, Debug, PartialEq)]
pub struct JobRequest {
    pub id: JobId,
    pub kind: &'static str,
    pub input: Vec<u8>,
    pub priority: u32,
}

/// Classes of jobs (decision 112), most urgent first: what a view or its
/// gameplay cannot do without (bakes, the coarsest tiles, the tiles a view
/// draws and shows, those under focus entities); gameplay's searches; and
/// the rest of what views load.
pub const URGENT_JOBS: u8 = 0;
pub const GAMEPLAY_JOBS: u8 = 1;
pub const VIEW_JOBS: u8 = 2;

/// A job's priority: its `class`, then its `place` among the class's jobs
/// (at most 2^24 - 1).
pub const fn job_priority(class: u8, place: u32) -> u32 {
    ((class as u32) << 24) | if place > 0xff_ffff { 0xff_ffff } else { place }
}

/// A job priority's class.
pub const fn job_class(priority: u32) -> u8 {
    (priority >> 24) as u8
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

/// Workers and their jobs (decision 110): workers running them, jobs
/// handed out and not done, jobs submitted, completed, failed and cancelled
/// while running (decision 112), keys shared and their bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct JobStats {
    pub workers: u32,
    pub running: u32,
    pub submitted: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub shares: u32,
    pub shared_bytes: u64,
}

/// Jobs handed to the host's workers (decision 110). Inserted by default,
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
        });
        id
    }

    /// Job `id`'s output once done, taken: `Err` when it failed.
    pub fn take(&mut self, id: JobId) -> Option<Result<Vec<u8>, String>> {
        self.done.remove(&id)
    }

    /// Drops job `id`: a worker running it stops at its next slice
    /// (decision 112), and its output, if any, is thrown away.
    pub fn cancel(&mut self, id: JobId) {
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
    /// `out` (decision 112): their workers should stop them.
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

    fn finish(&mut self, id: JobId, output: Result<Vec<u8>, String>) {
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
            ..self.stats
        }
    }
}
