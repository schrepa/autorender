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

use engine::host::{BakeJobStats, JobId, JobMessage, JobRequest, JobShare};
use engine::{Game, GameHost};
use engine_worker::{BAKE_CHILD_JOB, BakeAction, BakeCoordinator};

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
    // The worker, as a sampler samples it.
    #[cfg(feature = "profiling")]
    sampled: Option<engine_profiler::sampler::SampledThread>,
    thread: Option<JoinHandle<()>>,
    // Jobs in flight, each with its priority (lower first).
    jobs: HashMap<u64, u32>,
    broken: bool,
}

#[derive(Clone, Copy)]
enum Route {
    Logical(JobId),
    Child { root: u64, child: u64, bytes: usize },
}

struct BakeRoot {
    graph: BakeCoordinator,
    priority: u32,
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
    next_transport: u64,
    routes: HashMap<u64, Route>,
    bakes: HashMap<u64, BakeRoot>,
    bake_stats: Option<BakeJobStats>,
}

impl JobPool {
    /// `count` workers, with `cache` keeping their outputs.
    pub fn new(count: usize, cache: Option<JobCache>) -> Self {
        let (report, reports) = channel();
        let slots = (0..count)
            .map(|index| {
                let (orders, inbox) = channel();
                let report = report.clone();
                // Each worker registers itself for sampling as it starts.
                #[cfg(feature = "profiling")]
                let (register, registered) = channel();
                let thread = std::thread::Builder::new()
                    .name(format!("engine-jobs-{index}"))
                    .spawn(move || {
                        #[cfg(feature = "profiling")]
                        let _ = register.send(engine_profiler::sampler::SampledThread::current(
                            format!("Job worker {index}"),
                        ));
                        work(index, &inbox, &report);
                    })
                    .expect("threads start");
                Slot {
                    orders,
                    #[cfg(feature = "profiling")]
                    sampled: registered.recv().ok().flatten(),
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
            next_transport: 0,
            routes: HashMap::new(),
            bakes: HashMap::new(),
            bake_stats: None,
        }
    }

    /// The workers, as a sampler samples them.
    #[cfg(feature = "profiling")]
    pub fn sampled_threads(&self) -> Vec<engine_profiler::sampler::SampledThread> {
        self.slots
            .iter()
            .filter_map(|slot| slot.sampled.clone())
            .collect()
    }

    /// Workers able to run jobs.
    pub fn size(&self) -> usize {
        self.slots.iter().filter(|slot| !slot.broken).count()
    }

    /// Serves `host`'s jobs from now on.
    pub fn attach<G: Game>(&mut self, host: &mut GameHost<G>) {
        self.clear();
        self.bake_stats = None;
        host.report_bake_jobs(None);
        host.set_job_workers(self.size() as u32);
    }

    /// Hands workers what `host` made since the last exchange, and `host`
    /// what workers finished.
    pub fn exchange<G: Game>(&mut self, host: &mut GameHost<G>) {
        self.collect(host);
        for message in host.take_jobs() {
            match message {
                JobMessage::Cancel(id) => {
                    self.cancel(id.0);
                }
                JobMessage::Share(JobShare::Clear) => self.clear(),
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
        self.pump(host);
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
        if request.kind == "terrain_bake"
            && self.size() >= 2
            && let Some(epoch) = self.next_transport.checked_add(1)
        {
            self.next_transport = epoch;
            if let Ok(graph) = BakeCoordinator::start(&request.input, epoch) {
                self.bakes.insert(
                    request.id.0,
                    BakeRoot {
                        graph,
                        priority: request.priority,
                    },
                );
                self.bake_stats.get_or_insert_with(BakeJobStats::default);
                return;
            }
        }
        if let Err(error) = self.send(
            request.kind,
            request.input,
            request.priority,
            Route::Logical(request.id),
        ) {
            self.fail_logical(host, request.id.0, error);
        }
    }

    fn send(
        &mut self,
        kind: &'static str,
        input: Vec<u8>,
        priority: u32,
        route: Route,
    ) -> Result<(), String> {
        let id = self
            .next_transport
            .checked_add(1)
            .ok_or("worker transport ID overflow")?;
        self.next_transport = id;
        let slot = self
            .slots
            .iter_mut()
            .filter(|slot| !slot.broken)
            .min_by_key(|slot| {
                let urgent = slot.jobs.values().min().copied().unwrap_or(u32::MAX);
                (slot.jobs.len(), std::cmp::Reverse(urgent))
            });
        let Some(slot) = slot else {
            return Err("no workers".into());
        };
        slot.orders
            .send(Order::Run {
                id,
                kind,
                input,
                priority,
            })
            .map_err(|error| error.to_string())?;
        slot.jobs.insert(id, priority);
        self.routes.insert(id, route);
        Ok(())
    }

    fn cancel(&mut self, logical: u64) {
        if let Some(mut root) = self.bakes.remove(&logical) {
            root.graph.cancel();
            if let Ok(actions) = root.graph.ready(0) {
                for action in actions {
                    if let BakeAction::Unshare { key } = action {
                        self.broadcast(|| Order::Unshare(key.clone()));
                    }
                }
            }
        }
        let ids: Vec<_> = self
            .routes
            .iter()
            .filter_map(|(&id, route)| {
                let owner = match route {
                    Route::Logical(id) => id.0,
                    Route::Child { root, .. } => *root,
                };
                (owner == logical).then_some(id)
            })
            .collect();
        for id in ids {
            self.routes.remove(&id);
            for slot in &mut self.slots {
                if slot.jobs.remove(&id).is_some() {
                    let _ = slot.orders.send(Order::Cancel(id));
                }
            }
        }
        self.keys.remove(&logical);
        self.verifying.remove(&logical);
    }

    fn clear(&mut self) {
        self.bakes.clear();
        self.routes.clear();
        self.keys.clear();
        self.verifying.clear();
        for slot in &mut self.slots {
            slot.jobs.clear();
        }
        self.broadcast(|| Order::Clear);
    }

    fn finish_logical<G: Game>(&mut self, host: &mut GameHost<G>, id: u64, output: Vec<u8>) {
        let key = self.keys.remove(&id);
        if let Some(kept) = self.verifying.remove(&id) {
            if let (Some(cache), Some(key)) = (&mut self.cache, key)
                && kept != output
            {
                cache.mismatch(&key);
            }
            return;
        }
        if let (Some(cache), Some(key)) = (&mut self.cache, key) {
            cache.store(&key, &output);
        }
        host.complete_job(JobId(id), output);
    }

    fn fail_logical<G: Game>(&mut self, host: &mut GameHost<G>, id: u64, error: String) {
        let verifying = self.verifying.contains_key(&id);
        self.cancel(id);
        if !verifying {
            host.fail_job(JobId(id), error);
        }
    }

    /// Ordinary jobs already occupy their slots. Children use only the
    /// remaining window, bounded over all roots, ordered by priority/id.
    fn pump<G: Game>(&mut self, host: &mut GameHost<G>) {
        let mut roots: Vec<_> = self
            .bakes
            .iter()
            .map(|(&id, root)| (root.priority, id))
            .collect();
        roots.sort_unstable();
        for (priority, id) in roots {
            let occupied: usize = self.slots.iter().map(|slot| slot.jobs.len()).sum();
            let limit = self.size().saturating_mul(2).saturating_sub(occupied);
            let actions = self
                .bakes
                .get_mut(&id)
                .expect("listed root")
                .graph
                .ready(limit);
            let actions = match actions {
                Ok(actions) => actions,
                Err(error) => {
                    self.fail_logical(host, id, error);
                    continue;
                }
            };
            for action in actions {
                match action {
                    BakeAction::Share { key, bytes } => {
                        let copies = self.size() as u64;
                        if let Some(stats) = &mut self.bake_stats {
                            stats.sent_bytes += bytes.len() as u64 * copies;
                        }
                        let bytes: std::sync::Arc<[u8]> = bytes.into();
                        self.broadcast(|| Order::Share(key.clone(), bytes.clone()));
                    }
                    BakeAction::Unshare { key } => self.broadcast(|| Order::Unshare(key.clone())),
                    BakeAction::Run { id: child, input } => {
                        let bytes = input.capacity();
                        if let Some(stats) = &mut self.bake_stats {
                            stats.sent_bytes += input.len() as u64;
                        }
                        if let Err(error) = self.send(
                            BAKE_CHILD_JOB,
                            input,
                            priority,
                            Route::Child {
                                root: id,
                                child,
                                bytes,
                            },
                        ) {
                            self.fail_logical(host, id, error);
                            break;
                        }
                    }
                }
            }
            if let Some(output) = self
                .bakes
                .get_mut(&id)
                .and_then(|root| root.graph.take_result())
            {
                self.bakes.remove(&id);
                self.finish_logical(host, id, output);
            }
        }
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
                    let Some(route) = self.take_route(id) else {
                        continue;
                    };
                    match route {
                        Route::Logical(id) => self.finish_logical(host, id.0, output),
                        Route::Child { root, child, .. } => {
                            if let Some(stats) = &mut self.bake_stats {
                                stats.completed_children += 1;
                                stats.received_bytes += output.len() as u64;
                            }
                            if let Some(bake) = self.bakes.get_mut(&root)
                                && let Err(error) = bake.graph.complete(child, &output)
                            {
                                self.fail_logical(host, root, error);
                            }
                        }
                    }
                }
                Report::Part(id, part) => {
                    if let Some(Route::Logical(logical)) = self.routes.get(&id)
                        && !self.verifying.contains_key(&logical.0)
                    {
                        host.part_job(*logical, part);
                    }
                }
                Report::Failed(id, error) => {
                    if let Some(route) = self.take_route(id) {
                        let logical = match route {
                            Route::Logical(id) => id.0,
                            Route::Child { root, .. } => root,
                        };
                        self.fail_logical(host, logical, error);
                    }
                }
                Report::Broken(index, error) => {
                    eprintln!("A job worker broke and takes no more jobs: {error}");
                    let slot = &mut self.slots[index];
                    slot.broken = true;
                    let ids: Vec<_> = slot.jobs.drain().map(|(id, _)| id).collect();
                    for id in ids {
                        if let Some(route) = self.routes.remove(&id) {
                            let logical = match route {
                                Route::Logical(id) => id.0,
                                Route::Child { root, .. } => root,
                            };
                            self.fail_logical(host, logical, "worker unavailable".into());
                        }
                    }
                    host.set_job_workers(self.size() as u32);
                }
            }
        }
    }

    fn take_route(&mut self, id: u64) -> Option<Route> {
        let route = self.routes.remove(&id)?;
        for slot in &mut self.slots {
            slot.jobs.remove(&id);
        }
        Some(route)
    }

    /// Tells `host` the cache's counters when they change.
    fn report<G: Game>(&mut self, host: &mut GameHost<G>) {
        let workers = self.size() as u64;
        if let Some(stats) = &mut self.bake_stats {
            stats.active_roots = self.bakes.len() as u64;
            stats.children_in_flight = self
                .routes
                .values()
                .filter(|route| matches!(route, Route::Child { .. }))
                .count() as u64;
            let busy = self
                .slots
                .iter()
                .filter(|slot| {
                    slot.jobs
                        .keys()
                        .any(|id| matches!(self.routes.get(id), Some(Route::Child { .. })))
                })
                .count() as u64;
            stats.peak_busy_workers = stats.peak_busy_workers.max(busy);
            stats.private_share_bytes = self
                .bakes
                .values()
                .map(|root| root.graph.diagnostics().share_bytes as u64 * workers)
                .sum();
            stats.coordinator_bytes = self
                .bakes
                .values()
                .map(|root| {
                    let value = root.graph.diagnostics();
                    (value.buffer_bytes + value.share_bytes) as u64
                })
                .sum();
            stats.queued_bytes = self
                .routes
                .values()
                .map(|route| match route {
                    Route::Child { bytes, .. } => *bytes as u64,
                    _ => 0,
                })
                .sum();
            stats.peak_bytes = stats
                .peak_bytes
                .max(stats.private_share_bytes + stats.coordinator_bytes + stats.queued_bytes);
            host.report_bake_jobs(Some(*stats));
        }
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
