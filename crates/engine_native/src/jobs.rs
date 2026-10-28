use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use engine::host::{JobId, JobOrder, JobShare, policy};
use engine::{Game, GameHost};
use engine_worker::Report;

use crate::cook::PackWriter;
use crate::job_cache::{self, DEFAULT_BUDGET, JobStore, Source};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobOptions {
    pub workers: usize,
    pub cache: CacheOptions,
    pub packs: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheOptions {
    Off,
    Keep { budget: u64 },
    Verify,
}

impl Default for JobOptions {
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
    Read {
        id: JobId,
        key: String,
        sources: Vec<Source>,
    },
    Stop,
}

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
    #[cfg(feature = "profiling")]
    sampled: Option<engine_profiler::sampler::SampledThread>,
    thread: Option<JoinHandle<()>>,
}

pub struct JobPool {
    slots: Vec<Slot>,
    reports: Receiver<Reported>,
    store: Option<JobStore>,
    reported: Option<(u64, u64, u64, u64)>,
    cook: Option<PackWriter>,
}

impl JobPool {
    pub fn new(count: usize, store: Option<JobStore>) -> Self {
        let (report, reports) = channel();
        let slots = (0..count)
            .map(|index| {
                let (orders, inbox) = channel();
                let report = report.clone();
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

    #[cfg(feature = "profiling")]
    pub fn sampled_threads(&self) -> Vec<engine_profiler::sampler::SampledThread> {
        self.slots
            .iter()
            .filter_map(|slot| slot.sampled.clone())
            .collect()
    }

    pub fn size(&self) -> usize {
        self.slots.len()
    }

    pub fn attach<G: Game>(&mut self, host: &mut GameHost<G>) {
        host.set_job_workers(self.size() as u32);
        let cache = self.store.as_ref().map(JobStore::verify);
        host.set_job_cache(cache.is_some(), cache == Some(true));
    }

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

    pub fn packed(&self) -> usize {
        self.store.as_ref().map_or(0, JobStore::packed)
    }

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

fn work(index: u32, inbox: &Receiver<Order>, report: &Sender<Reported>) {
    let mut worker = engine_worker::Worker::new();
    loop {
        let mut next = if worker.busy() {
            inbox.try_recv().ok()
        } else {
            match inbox.recv() {
                Ok(order) => Some(order),
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
