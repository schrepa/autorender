//! Worker jobs natively, as the page runs them: threads beside the game,
//! each holding the engine's job worker, which runs its most urgent job a
//! slice at a time (`engine_worker::Worker::slice`). The engine schedules
//! its jobs (`engine::host::JobScheduler`): the pool carries its orders to
//! the workers and to the job cache's folder, and hands back what workers
//! report and the cache reads. A worker's panic ends the program, as every
//! engine panic does (`engine_panic`'s hook).

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use engine::host::{JobId, JobOrder, JobShare, policy};
use engine::{Game, GameHost};
use engine_worker::Report;

use crate::job_cache::{DEFAULT_BUDGET, JobStore};

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
    /// The hosts' default workers for the logical processors, and kept
    /// outputs ([`engine::host::policy`]).
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map_or(4, usize::from);
        Self {
            workers: policy::default_workers(u32::try_from(cores).unwrap_or(u32::MAX)) as usize,
            cache: CacheOptions::Keep {
                budget: DEFAULT_BUDGET,
            },
        }
    }
}

impl JobOptions {
    /// The pool these options run; none without workers.
    pub fn pool(self) -> Option<JobPool> {
        let store = match self.cache {
            CacheOptions::Off => None,
            CacheOptions::Keep { budget } => JobStore::open(budget, false),
            CacheOptions::Verify => JobStore::open(DEFAULT_BUDGET, true),
        };
        (self.workers > 0).then(|| JobPool::new(self.workers, store))
    }
}

/// What the pool tells a worker, in order.
enum Order {
    Share(String, Arc<[u8]>),
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

struct Slot {
    orders: Sender<Order>,
    // The worker, as a sampler samples it.
    #[cfg(feature = "profiling")]
    sampled: Option<engine_profiler::sampler::SampledThread>,
    thread: Option<JoinHandle<()>>,
}

/// Job workers beside a game, and the folder keeping their outputs.
pub struct JobPool {
    slots: Vec<Slot>,
    reports: Receiver<Report>,
    store: Option<JobStore>,
    reported: Option<(u64, u64, u64, u64)>,
}

impl JobPool {
    /// `count` workers, with `store` keeping their outputs.
    pub fn new(count: usize, store: Option<JobStore>) -> Self {
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
                        work(&inbox, &report);
                    })
                    .expect("threads start");
                Slot {
                    orders,
                    #[cfg(feature = "profiling")]
                    sampled: registered.recv().ok().flatten(),
                    thread: Some(thread),
                }
            })
            .collect();
        Self {
            slots,
            reports,
            store,
            reported: None,
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

    /// Workers that run jobs.
    pub fn size(&self) -> usize {
        self.slots.len()
    }

    /// Serves `host`'s jobs from now on.
    pub fn attach<G: Game>(&mut self, host: &mut GameHost<G>) {
        host.set_job_workers(self.size() as u32);
        let cache = self.store.as_ref().map(JobStore::verify);
        host.set_job_cache(cache.is_some(), cache == Some(true));
    }

    /// Hands `host` what workers finished since the last exchange, and
    /// carries out its orders since.
    pub fn exchange<G: Game>(&mut self, host: &mut GameHost<G>) {
        while let Ok(report) = self.reports.try_recv() {
            match report {
                Report::Done(id, output) => host.complete_job(JobId(id), output),
                Report::Part(id, part) => host.part_job(JobId(id), part),
                Report::Failed(id, error) => host.fail_job(JobId(id), error),
            }
        }
        // Reads from the cache answer at once, and may order more.
        let mut orders = host.take_jobs();
        while !orders.is_empty() {
            for order in orders {
                self.carry(host, order);
            }
            orders = host.take_jobs();
        }
        if let Some(store) = &self.store {
            let counters = store.counters();
            if self.reported != Some(counters) {
                self.reported = Some(counters);
                let (hits, misses, bytes, mismatches) = counters;
                host.report_job_cache(hits, misses, bytes, mismatches);
            }
        }
    }

    /// Carries `order` to the workers or the job cache's folder.
    fn carry<G: Game>(&mut self, host: &mut GameHost<G>, order: JobOrder) {
        match order {
            JobOrder::Share(JobShare::Clear) => self.broadcast(|| Order::Clear),
            JobOrder::Share(JobShare::Set { key, bytes }) => {
                let bytes: Arc<[u8]> = bytes.into();
                self.broadcast(|| Order::Share(key.clone(), bytes.clone()));
            }
            JobOrder::Share(JobShare::Remove { key }) => {
                self.broadcast(|| Order::Unshare(key.clone()));
            }
            JobOrder::Run { worker, request } => {
                let _ = self.slots[worker as usize].orders.send(Order::Run {
                    id: request.id.0,
                    kind: request.kind,
                    input: request.input,
                    priority: request.priority,
                });
            }
            JobOrder::Cancel { worker, id } => {
                let _ = self.slots[worker as usize].orders.send(Order::Cancel(id.0));
            }
            JobOrder::Lookup { id, kind, identity } => {
                let Some(store) = &mut self.store else {
                    host.job_kept(id, String::new(), None);
                    return;
                };
                let (key, kept) = store.lookup(kind, &identity);
                host.job_kept(id, key, kept);
            }
            JobOrder::Keep { key, output } => {
                if let Some(store) = &mut self.store {
                    store.keep(&key, &output);
                }
            }
            JobOrder::Mismatch { key } => {
                if let Some(store) = &mut self.store {
                    store.mismatch(&key);
                }
            }
        }
    }

    fn broadcast(&mut self, order: impl Fn() -> Order) {
        for slot in &self.slots {
            let _ = slot.orders.send(order());
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
        if let Some(store) = &mut self.store {
            store.save();
        }
    }
}

/// A worker's loop: its orders in turn, and between them a slice of its
/// most urgent job; reports to `report`.
fn work(inbox: &Receiver<Order>, report: &Sender<Report>) {
    let mut worker = engine_worker::Worker::new();
    worker.warm();
    loop {
        // Orders first: waiting for one while no job is held.
        let mut next = if worker.busy() {
            inbox.try_recv().ok()
        } else {
            match inbox.recv() {
                Ok(order) => Some(order),
                // The pool is gone.
                Err(_) => return,
            }
        };
        while let Some(order) = next.take() {
            match order {
                Order::Share(key, bytes) => {
                    if let Err(error) = worker.share(&key, &bytes) {
                        eprintln!("A job worker refused share {key}: {error}");
                    }
                }
                Order::Unshare(key) => worker.unshare(&key),
                Order::Clear => worker.clear(),
                Order::Run {
                    id,
                    kind,
                    input,
                    priority,
                } => worker.hold(id, kind, input, priority),
                Order::Cancel(id) => worker.cancel(id),
                Order::Stop => return,
            }
            next = inbox.try_recv().ok();
        }
        if let Some(sliced) = worker.slice() {
            let _ = report.send(sliced);
        }
    }
}
