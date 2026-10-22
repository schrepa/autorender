//! Worker jobs natively, as the page runs them: threads beside the game,
//! each holding the engine's job worker, which runs its most urgent job a
//! slice at a time (`engine_worker::Worker::slice`). The engine schedules
//! its jobs (`engine::host::JobScheduler`): the pool carries its orders to
//! the workers and to the records (the cook's packs, then the store's
//! folder), and hands back what workers report and the records read. A
//! record looked up is read and unpacked on the worker the engine waits
//! least on, as the page's workers unpack theirs, and comes back with the
//! workers' reports; the game's thread only finds where it lies and counts
//! it. A pool without workers has the host run its jobs, and read its
//! records, in place. Cooking, it keeps every record the session read or
//! made for the pack it writes. A worker's panic ends the program, as every
//! engine panic does (`engine_panic`'s hook).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use engine::host::{JobId, JobOrder, JobShare, policy};
use engine::{Game, GameHost};
use engine_worker::Report;

use crate::cook::PackWriter;
use crate::job_cache::{self, DEFAULT_BUDGET, JobStore, Source};

/// How a native host runs the engine's jobs, as the page's `workers` and
/// `jobcache` parameters choose, and where its records come from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobOptions {
    /// Worker threads; with none, the host runs jobs in place.
    pub workers: usize,
    pub cache: CacheOptions,
    /// The folder of the cook's packs it reads (`--packs`).
    pub packs: Option<PathBuf>,
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
            packs: None,
        }
    }
}

impl JobOptions {
    /// The pool these options run, cooking where `cook` says: its records
    /// then come from a store whatever the cache's options.
    pub fn pool(&self, cook: bool) -> JobPool {
        let packs = self.packs.as_deref();
        let store = match self.cache {
            CacheOptions::Off if cook => JobStore::open(DEFAULT_BUDGET, false, packs),
            CacheOptions::Off => None,
            CacheOptions::Keep { budget } => JobStore::open(budget, false, packs),
            CacheOptions::Verify => JobStore::open(DEFAULT_BUDGET, true, packs),
        };
        let mut pool = JobPool::new(self.workers, store);
        if cook {
            pool.cook = Some(PackWriter::default());
        }
        pool
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
    /// Read the record `key` lookup `id` asks for from `sources`, in order.
    Read {
        id: JobId,
        key: String,
        sources: Vec<Source>,
    },
    Stop,
}

/// What a worker hands back: a slice's report, or what reading a record
/// looked up came to on worker `worker` ([`job_cache::read`]).
enum Reported {
    Job(Report),
    Read {
        worker: u32,
        id: JobId,
        key: String,
        sources: Vec<Source>,
        read: Option<(usize, Vec<u8>)>,
    },
}

struct Slot {
    orders: Sender<Order>,
    // The worker, as a sampler samples it.
    #[cfg(feature = "profiling")]
    sampled: Option<engine_profiler::sampler::SampledThread>,
    thread: Option<JoinHandle<()>>,
}

/// Job workers beside a game, its records, and while cooking what the
/// pack takes.
pub struct JobPool {
    slots: Vec<Slot>,
    reports: Receiver<Reported>,
    store: Option<JobStore>,
    reported: Option<(u64, u64, u64, u64)>,
    cook: Option<PackWriter>,
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
                        work(index as u32, &inbox, &report);
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
            cook: None,
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

    /// Hands `host` what workers finished and the records they read since
    /// the last exchange, and carries out its orders since.
    pub fn exchange<G: Game>(&mut self, host: &mut GameHost<G>) {
        while let Ok(reported) = self.reports.try_recv() {
            match reported {
                Reported::Job(Report::Done(id, output)) => host.complete_job(JobId(id), output),
                Reported::Job(Report::Failed(id, error)) => host.fail_job(JobId(id), error),
                Reported::Read {
                    worker,
                    id,
                    key,
                    sources,
                    read,
                } => {
                    host.job_task_done(worker);
                    self.answer(host, (id, key), &sources, read);
                }
            }
        }
        // Records read in place answer at once, and may order more.
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
            JobOrder::Lookup { id, key } => {
                let sources = self
                    .store
                    .as_mut()
                    .map(|store| store.sources(&key))
                    .unwrap_or_default();
                if sources.is_empty() {
                    host.job_kept(id, key, None);
                    return;
                }
                // Its bytes are read and unpacked on a worker, without one
                // here.
                match host.job_task() {
                    Some(worker) => {
                        let _ = self.slots[worker as usize].orders.send(Order::Read {
                            id,
                            key,
                            sources,
                        });
                    }
                    None => {
                        let read = job_cache::read(&sources);
                        self.answer(host, (id, key), &sources, read);
                    }
                }
            }
            JobOrder::Keep { key, output } => {
                if let Some(cook) = &mut self.cook {
                    cook.take(&key, &output);
                }
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

    /// Hands `host` the record `key` lookup `id` asked for, as reading it
    /// from `sources` came to (`read`) and the store counted it; the cook's
    /// pack takes what it held.
    fn answer<G: Game>(
        &mut self,
        host: &mut GameHost<G>,
        (id, key): (JobId, String),
        sources: &[Source],
        read: Option<(usize, Vec<u8>)>,
    ) {
        let kept = self
            .store
            .as_mut()
            .and_then(|store| store.answered(&key, sources, read));
        if let (Some(cook), Some(record)) = (&mut self.cook, &kept) {
            cook.take(&key, record);
        }
        host.job_kept(id, key, kept);
    }

    /// The records the packs it reads hold.
    pub fn packed(&self) -> usize {
        self.store.as_ref().map_or(0, JobStore::packed)
    }

    /// Writes what the cooking session read or made as the pack `path`,
    /// once the packs it reads are closed (it may replace one): the
    /// records and bytes written.
    pub fn write_pack(&mut self, path: &Path) -> Result<(usize, u64), String> {
        let cook = self.cook.take().ok_or("the pool is not cooking")?;
        if let Some(store) = &mut self.store {
            store.close_packs();
        }
        let bytes = cook.write(path)?;
        Ok((cook.len(), bytes))
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

/// Worker `index`'s loop: its orders in turn (records read for lookups at
/// once), and between them a slice of its most urgent job; reports to
/// `report`.
fn work(index: u32, inbox: &Receiver<Order>, report: &Sender<Reported>) {
    let mut worker = engine_worker::Worker::new();
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
                    if let Err(error) = worker.share(&key, bytes) {
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
                Order::Read { id, key, sources } => {
                    let read = job_cache::read(&sources);
                    let _ = report.send(Reported::Read {
                        worker: index,
                        id,
                        key,
                        sources,
                        read,
                    });
                }
                Order::Stop => return,
            }
            next = inbox.try_recv().ok();
        }
        if let Some(sliced) = worker.slice() {
            let _ = report.send(Reported::Job(sliced));
        }
    }
}
