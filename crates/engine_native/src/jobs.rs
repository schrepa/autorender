//! Worker jobs natively, as the page runs them: threads beside the game,
//! each holding the engine's job worker. The engine hands the host, in
//! order, what every worker should hold (terrain stacks and recipes) and
//! jobs (a terrain tile, a walk, a bake); the pool hands each share to
//! every worker and each job to the worker with the fewest in flight, among
//! those the one whose most urgent job is least urgent, and hands outputs
//! back as workers finish them. Workers run their most urgent job a slice at
//! a time; jobs the engine cancels stop at their worker's next slice, and
//! parts of outputs ready early, as a tile's shape, go back as they come.
//! Outputs of jobs the engine marks keepable come from the job cache where
//! the same build made them before. A worker that panics takes no more
//! jobs, and the engine takes back those it held.

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use engine::host::{JobId, JobMessage, JobRequest, JobShare};
use engine::{Game, GameHost};

use crate::job_cache::{DEFAULT_BUDGET, JobCache};

/// How a native host runs the engine's jobs, as the page's `workers` and
/// `jobcache` parameters choose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JobOptions {
    /// Worker threads; with none, the engine runs its jobs itself.
    pub workers: usize,
    pub cache: CacheOptions,
}

/// What the job cache keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheOptions {
    Off,
    /// Outputs up to `budget` bytes, the least recently used going first.
    Keep {
        budget: u64,
    },
    /// Every job runs anyway, its output compared with the one kept.
    Verify,
}

impl Default for JobOptions {
    /// Half the logical processors, at most 8, as the page runs, and 1 GiB
    /// of kept outputs.
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map_or(4, usize::from);
        Self {
            workers: (cores / 2).clamp(1, 8),
            cache: CacheOptions::Keep {
                budget: DEFAULT_BUDGET,
            },
        }
    }
}

impl JobOptions {
    /// The pool these options run; none without workers.
    pub fn pool(self) -> Option<JobPool> {
        let cache = match self.cache {
            CacheOptions::Off => None,
            CacheOptions::Keep { budget } => JobCache::open(budget, false),
            CacheOptions::Verify => JobCache::open(DEFAULT_BUDGET, true),
        };
        (self.workers > 0).then(|| JobPool::new(self.workers, cache))
    }
}

// A slice of work between a worker's looks at its messages.
const SLICE: Duration = Duration::from_millis(8);
const STEP_UNITS: u64 = 4096;

/// What the pool tells a worker, in order.
enum Order {
    Share(String, std::sync::Arc<[u8]>),
    Unshare(String),
    Clear,
    Run {
        id: u64,
        kind: &'static str,
        input: Vec<u8>,
        priority: u32,
    },
    Cancel(u64),
    Stop,
}

/// What a worker tells the pool.
enum Report {
    Done(u64, Vec<u8>),
    Part(u64, Vec<u8>),
    Failed(u64, String),
    /// The worker panicked: it takes no more jobs.
    Broken(usize, String),
}

struct Slot {
    orders: Sender<Order>,
    thread: Option<JoinHandle<()>>,
    // Jobs in flight, each with its priority (lower first).
    jobs: HashMap<u64, u32>,
    broken: bool,
}

/// Job workers beside a game, and the cache of their outputs.
pub struct JobPool {
    slots: Vec<Slot>,
    reports: Receiver<Report>,
    cache: Option<JobCache>,
    // The keys of keepable jobs in flight, and outputs kept that fresh
    // runs verify.
    keys: HashMap<u64, String>,
    verifying: HashMap<u64, Vec<u8>>,
    reported: Option<(u64, u64, u64, u64)>,
}

impl JobPool {
    /// `count` workers, with `cache` keeping their outputs.
    pub fn new(count: usize, cache: Option<JobCache>) -> Self {
        let (report, reports) = channel();
        let slots = (0..count)
            .map(|index| {
                let (orders, inbox) = channel();
                let report = report.clone();
                let thread = std::thread::Builder::new()
                    .name(format!("engine-jobs-{index}"))
                    .spawn(move || work(index, &inbox, &report))
                    .expect("threads start");
                Slot {
                    orders,
                    thread: Some(thread),
                    jobs: HashMap::new(),
                    broken: false,
                }
            })
            .collect();
        Self {
            slots,
            reports,
            cache,
            keys: HashMap::new(),
            verifying: HashMap::new(),
            reported: None,
        }
    }

    /// Workers able to run jobs.
    pub fn size(&self) -> usize {
        self.slots.iter().filter(|slot| !slot.broken).count()
    }

    /// Serves `host`'s jobs from now on.
    pub fn attach<G: Game>(&mut self, host: &mut GameHost<G>) {
        host.set_job_workers(self.size() as u32);
    }

    /// Hands workers what `host` made since the last exchange, and `host`
    /// what workers finished.
    pub fn exchange<G: Game>(&mut self, host: &mut GameHost<G>) {
        self.collect(host);
        for message in host.take_jobs() {
            match message {
                JobMessage::Cancel(id) => {
                    self.keys.remove(&id.0);
                    self.verifying.remove(&id.0);
                    if let Some(slot) = self
                        .slots
                        .iter_mut()
                        .find(|slot| slot.jobs.contains_key(&id.0))
                    {
                        slot.jobs.remove(&id.0);
                        let _ = slot.orders.send(Order::Cancel(id.0));
                    }
                }
                JobMessage::Share(JobShare::Clear) => self.broadcast(|| Order::Clear),
                JobMessage::Share(JobShare::Set { key, bytes }) => {
                    let bytes: std::sync::Arc<[u8]> = bytes.into();
                    self.broadcast(|| Order::Share(key.clone(), bytes.clone()));
                }
                JobMessage::Share(JobShare::Remove { key }) => {
                    self.broadcast(|| Order::Unshare(key.clone()));
                }
                JobMessage::Run(request) => self.request(host, request),
            }
        }
        self.report(host);
    }

    /// Answers a keepable job from the cache, or hands it to a worker and
    /// keeps its output once done; verifying, runs it anyway and compares.
    fn request<G: Game>(&mut self, host: &mut GameHost<G>, request: JobRequest) {
        if let (Some(cache), Some(identity)) = (&mut self.cache, &request.identity) {
            let key = cache.key(request.kind, identity);
            if let Some(kept) = cache.lookup(&key) {
                if !cache.verify() {
                    host.complete_job(request.id, kept);
                    return;
                }
                host.complete_job(request.id, kept.clone());
                self.verifying.insert(request.id.0, kept);
            }
            self.keys.insert(request.id.0, key);
        }
        self.dispatch(host, request);
    }

    /// Hands `request` to the worker with the fewest jobs in flight; among
    /// those, the one whose most urgent job is least urgent.
    fn dispatch<G: Game>(&mut self, host: &mut GameHost<G>, request: JobRequest) {
        let slot = self
            .slots
            .iter_mut()
            .filter(|slot| !slot.broken)
            .min_by_key(|slot| {
                let urgent = slot.jobs.values().min().copied().unwrap_or(u32::MAX);
                (slot.jobs.len(), std::cmp::Reverse(urgent))
            });
        let Some(slot) = slot else {
            host.fail_job(request.id, "no workers".into());
            return;
        };
        slot.jobs.insert(request.id.0, request.priority);
        let _ = slot.orders.send(Order::Run {
            id: request.id.0,
            kind: request.kind,
            input: request.input,
            priority: request.priority,
        });
    }

    fn broadcast(&mut self, order: impl Fn() -> Order) {
        for slot in self.slots.iter().filter(|slot| !slot.broken) {
            let _ = slot.orders.send(order());
        }
    }

    /// Hands `host` what workers finished since the last call.
    fn collect<G: Game>(&mut self, host: &mut GameHost<G>) {
        while let Ok(report) = self.reports.try_recv() {
            match report {
                Report::Done(id, output) => {
                    // Jobs cancelled since are no one's.
                    let Some(slot) = self
                        .slots
                        .iter_mut()
                        .find(|slot| slot.jobs.contains_key(&id))
                    else {
                        continue;
                    };
                    slot.jobs.remove(&id);
                    let key = self.keys.remove(&id);
                    // A kept output the engine already took is compared,
                    // not taken.
                    if let Some(kept) = self.verifying.remove(&id) {
                        if let (Some(cache), Some(key)) = (&mut self.cache, key)
                            && kept != output
                        {
                            cache.mismatch(&key);
                        }
                        continue;
                    }
                    if let (Some(cache), Some(key)) = (&mut self.cache, key) {
                        cache.store(&key, &output);
                    }
                    host.complete_job(JobId(id), output);
                }
                Report::Part(id, part) => {
                    if self.slots.iter().any(|slot| slot.jobs.contains_key(&id))
                        && !self.verifying.contains_key(&id)
                    {
                        host.part_job(JobId(id), part);
                    }
                }
                Report::Failed(id, error) => {
                    let Some(slot) = self
                        .slots
                        .iter_mut()
                        .find(|slot| slot.jobs.contains_key(&id))
                    else {
                        continue;
                    };
                    slot.jobs.remove(&id);
                    self.keys.remove(&id);
                    if self.verifying.remove(&id).is_none() {
                        host.fail_job(JobId(id), error);
                    }
                }
                Report::Broken(index, error) => {
                    eprintln!("A job worker broke and takes no more jobs: {error}");
                    let slot = &mut self.slots[index];
                    slot.broken = true;
                    for (id, _) in slot.jobs.drain() {
                        self.keys.remove(&id);
                        if self.verifying.remove(&id).is_none() {
                            host.fail_job(JobId(id), "worker unavailable".into());
                        }
                    }
                    host.set_job_workers(self.size() as u32);
                }
            }
        }
    }

    /// Tells `host` the cache's counters when they change.
    fn report<G: Game>(&mut self, host: &mut GameHost<G>) {
        let Some(cache) = &self.cache else {
            return;
        };
        let counters = cache.counters();
        if self.reported != Some(counters) {
            self.reported = Some(counters);
            let (hits, misses, bytes, mismatches) = counters;
            host.report_job_cache(hits, misses, bytes, mismatches);
        }
    }
}

impl Drop for JobPool {
    fn drop(&mut self) {
        for slot in &self.slots {
            let _ = slot.orders.send(Order::Stop);
        }
        for slot in &mut self.slots {
            if let Some(thread) = slot.thread.take() {
                let _ = thread.join();
            }
        }
        if let Some(cache) = &mut self.cache {
            cache.save();
        }
    }
}

/// A job a worker holds.
struct Held {
    id: u64,
    kind: &'static str,
    input: Vec<u8>,
    priority: u32,
    started: bool,
}

/// Worker `index`'s loop: its orders in turn, and between them its most
/// urgent job a slice at a time; reports to `report`.
fn work(index: usize, inbox: &Receiver<Order>, report: &Sender<Report>) {
    let mut worker = engine_worker::Worker::new();
    worker.warm();
    let mut jobs: Vec<Held> = Vec::new();
    loop {
        // Orders first: waiting for one while no job is held.
        let mut next = if jobs.is_empty() {
            match inbox.recv() {
                Ok(order) => Some(order),
                // The pool is gone.
                Err(_) => return,
            }
        } else {
            inbox.try_recv().ok()
        };
        while let Some(order) = next.take() {
            match order {
                Order::Share(key, bytes) => {
                    if let Err(error) = worker.share(&key, &bytes) {
                        eprintln!("A job worker refused share {key}: {error}");
                    }
                }
                Order::Unshare(key) => worker.unshare(&key),
                Order::Clear => {
                    worker.clear();
                    jobs.clear();
                }
                Order::Run {
                    id,
                    kind,
                    input,
                    priority,
                } => jobs.push(Held {
                    id,
                    kind,
                    input,
                    priority,
                    started: false,
                }),
                Order::Cancel(id) => {
                    jobs.retain(|job| job.id != id);
                    worker.cancel(id);
                }
                Order::Stop => return,
            }
            next = inbox.try_recv().ok();
        }
        // The most urgent job: lowest priority, then first handed.
        let Some(at) = (0..jobs.len()).min_by_key(|&at| (jobs[at].priority, jobs[at].id)) else {
            continue;
        };
        let job = &mut jobs[at];
        let slice = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if !job.started {
                worker.start(job.id, job.kind, &job.input)?;
                job.started = true;
            }
            let start = Instant::now();
            loop {
                if let Some(output) = worker.step(job.id, STEP_UNITS)? {
                    return Ok(Some(output));
                }
                if start.elapsed() >= SLICE {
                    return Ok(None);
                }
            }
        }));
        let id = job.id;
        match slice {
            Ok(Ok(Some(output))) => {
                jobs.remove(at);
                let _ = report.send(Report::Done(id, output));
            }
            Ok(Ok(None)) => {
                if let Some(part) = worker.take_part(id) {
                    let _ = report.send(Report::Part(id, part));
                }
            }
            Ok(Err(error)) => {
                jobs.remove(at);
                worker.cancel(id);
                let _ = report.send(Report::Failed(id, error));
            }
            Err(panic) => {
                let error = panic
                    .downcast_ref::<&str>()
                    .map(|text| (*text).to_owned())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panicked".into());
                let _ = report.send(Report::Broken(index, error));
                return;
            }
        }
    }
}
