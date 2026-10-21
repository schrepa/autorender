//! The engine's jobs as every host runs them on its workers
//! ([`JobScheduler`]): which worker takes each job, among those with the
//! fewest in flight the one whose most urgent job is least urgent; what
//! every worker holds; which records the host's packs and store answer, and
//! which a fresh run verifies; which jobs run in place, on a host without
//! workers; and what a new engine's first message drops. Hosts carry its
//! orders ([`JobOrder`]) to their workers and their records (the cook's
//! packs, `engine_worker::JobCache` and the bytes they keep), and hand back
//! what workers report and the records read.

use engine_core::collections::HashMap;
use engine_core::{JobId, JobRequest, JobShare, MISSING_RECORD, RecordKey};

use crate::app_host::JobMessage;

/// What a host does for the engine's jobs, in order.
#[derive(Debug)]
pub enum JobOrder {
    /// Every worker holds bytes as a key, drops a key, or drops everything
    /// (a new engine's first order: the earlier engine's jobs are no one's).
    Share(JobShare),
    /// Worker `worker` runs `request`, its most urgent job first.
    Run { worker: u32, request: JobRequest },
    /// Worker `worker` stops job `id`.
    Cancel { worker: u32, id: JobId },
    /// The host's packs, then its store, look up the record `key` (64
    /// hexadecimal digits) and hand back what they hold, if anything, for
    /// job `id` ([`crate::GameHost::job_kept`]).
    Lookup { id: JobId, key: String },
    /// The store keeps `output` as the record `key`.
    Keep { key: String, output: Vec<u8> },
    /// A fresh run's output differed from the record kept as `key`, which
    /// goes.
    Mismatch { key: String },
}

/// How a job ended for the engine: its output, or why it failed.
pub(crate) type Finished = (JobId, Result<Vec<u8>, String>);

/// A worker's load: its jobs in flight with their priorities (lower
/// first), and the host's own tasks on it.
#[derive(Debug, Default)]
struct Load {
    jobs: HashMap<u64, u32>,
    tasks: u32,
    // Broken and not replaced: it takes nothing more.
    lost: bool,
}

/// What a lookup is for: a job, run where no record answers it, or a
/// record fetched alone.
#[derive(Debug)]
enum Looking {
    Job(JobRequest),
    Fetch,
}

/// The engine's jobs on a host's workers.
#[derive(Debug, Default)]
pub struct JobScheduler {
    workers: Vec<Load>,
    // The host holds records (packs, a store), and verifies them.
    caching: bool,
    verify: bool,
    // Each job in flight: the worker running it.
    running: HashMap<u64, u32>,
    // Jobs making a record, in flight: the key it is kept as.
    keys: HashMap<u64, String>,
    // Records that a fresh run verifies, the engine having taken them.
    verifying: HashMap<u64, Vec<u8>>,
    // Lookups in flight.
    looking: HashMap<u64, Looking>,
    // Jobs no worker takes, which the host runs in place.
    here: Vec<JobRequest>,
    // Orders made between the host's takes.
    orders: Vec<JobOrder>,
}

impl JobScheduler {
    /// `workers` workers, those already running kept.
    pub(crate) fn set_workers(&mut self, workers: u32) {
        self.workers.resize_with(workers as usize, Load::default);
    }

    /// Workers that take jobs.
    pub(crate) fn working(&self) -> u32 {
        self.workers.iter().filter(|load| !load.lost).count() as u32
    }

    /// Whether the host holds records, and verifies them.
    pub(crate) fn set_cache(&mut self, caching: bool, verify: bool) {
        self.caching = caching;
        self.verify = caching && verify;
    }

    /// What the host does for the engine's `messages` (its jobs since the
    /// last take, in order), after the orders made since; records fetched
    /// that no host holds fail into `finished`.
    pub(crate) fn take(
        &mut self,
        messages: Vec<JobMessage>,
        finished: &mut Vec<Finished>,
    ) -> Vec<JobOrder> {
        for message in messages {
            match message {
                JobMessage::Cancel(id) => {
                    self.looking.remove(&id.0);
                    self.keys.remove(&id.0);
                    self.verifying.remove(&id.0);
                    self.here.retain(|request| request.id != id);
                    if let Some(worker) = self.running.remove(&id.0) {
                        self.workers[worker as usize].jobs.remove(&id.0);
                        self.orders.push(JobOrder::Cancel { worker, id });
                    }
                }
                JobMessage::Share(JobShare::Clear) => {
                    for load in &mut self.workers {
                        load.jobs.clear();
                    }
                    self.running.clear();
                    self.keys.clear();
                    self.verifying.clear();
                    self.looking.clear();
                    self.here.clear();
                    self.orders.push(JobOrder::Share(JobShare::Clear));
                }
                JobMessage::Share(share) => self.orders.push(JobOrder::Share(share)),
                JobMessage::Run(request) => match request.record {
                    Some(record) if self.caching => {
                        self.lookup(request.id, record, Looking::Job(request));
                    }
                    _ => self.dispatch(request),
                },
                JobMessage::Fetch(fetch) if self.caching => {
                    self.lookup(fetch.id, fetch.record, Looking::Fetch);
                }
                JobMessage::Fetch(fetch) => {
                    finished.push((fetch.id, Err(MISSING_RECORD.to_owned())));
                }
                JobMessage::Keep(record, output) => {
                    if self.caching {
                        self.orders.push(JobOrder::Keep {
                            key: record.hex(),
                            output,
                        });
                    }
                }
            }
        }
        std::mem::take(&mut self.orders)
    }

    fn lookup(&mut self, id: JobId, record: RecordKey, looking: Looking) {
        self.orders.push(JobOrder::Lookup {
            id,
            key: record.hex(),
        });
        self.looking.insert(id.0, looking);
    }

    /// The host's packs and store looked up job `id`'s record, `key`: the
    /// bytes they hold, if any, are the job's, and verifying, the job runs
    /// anyway and compares; without them, the job runs and the store keeps
    /// its output. A record fetched alone that none holds fails.
    pub(crate) fn kept(
        &mut self,
        id: JobId,
        key: String,
        output: Option<Vec<u8>>,
        finished: &mut Vec<Finished>,
    ) {
        // Cancelled or cleared meanwhile: no one's.
        let Some(looking) = self.looking.remove(&id.0) else {
            return;
        };
        let request = match looking {
            Looking::Fetch => {
                finished.push((id, output.ok_or_else(|| MISSING_RECORD.to_owned())));
                return;
            }
            Looking::Job(request) => request,
        };
        match output {
            Some(output) if self.verify => {
                finished.push((id, Ok(output.clone())));
                self.verifying.insert(id.0, output);
            }
            Some(output) => {
                finished.push((id, Ok(output)));
                return;
            }
            None => {}
        }
        self.keys.insert(id.0, key);
        self.dispatch(request);
    }

    /// Hands `request` to the worker with the fewest jobs and tasks in
    /// flight; among those, the one whose most urgent job is least urgent,
    /// where it would wait least. Without one, the host runs it in place.
    fn dispatch(&mut self, request: JobRequest) {
        let Some(worker) = self.least() else {
            self.here.push(request);
            return;
        };
        self.workers[worker as usize]
            .jobs
            .insert(request.id.0, request.priority);
        self.running.insert(request.id.0, worker);
        self.orders.push(JobOrder::Run { worker, request });
    }

    /// The jobs no worker takes, for the host to run in place
    /// ([`Self::ran_here`]), in order.
    pub(crate) fn take_here(&mut self) -> Vec<JobRequest> {
        std::mem::take(&mut self.here)
    }

    /// The worker a new job or task waits least on.
    fn least(&self) -> Option<u32> {
        self.workers
            .iter()
            .enumerate()
            .filter(|(_, load)| !load.lost)
            .min_by_key(|(_, load)| {
                let urgent = load.jobs.values().min().copied().unwrap_or(u32::MAX);
                (
                    load.jobs.len() + load.tasks as usize,
                    std::cmp::Reverse(urgent),
                )
            })
            .map(|(worker, _)| worker as u32)
    }

    /// Job `id` is done on its worker: `output` is the engine's, which the
    /// store keeps where the job makes a record; verifying, it is compared
    /// with the record, which the engine took.
    pub(crate) fn done(&mut self, id: JobId, output: Vec<u8>, finished: &mut Vec<Finished>) {
        // Jobs cancelled since are no one's.
        let Some(worker) = self.running.remove(&id.0) else {
            return;
        };
        self.workers[worker as usize].jobs.remove(&id.0);
        self.made(id, Ok(output), finished);
    }

    /// Job `id`, which the host ran in place, came to `output`: as
    /// [`Self::done`] and [`Self::failed`] take a worker's.
    pub(crate) fn ran_here(
        &mut self,
        id: JobId,
        output: Result<Vec<u8>, String>,
        finished: &mut Vec<Finished>,
    ) {
        self.made(id, output, finished);
    }

    fn made(&mut self, id: JobId, output: Result<Vec<u8>, String>, finished: &mut Vec<Finished>) {
        let key = self.keys.remove(&id.0);
        if let Some(kept) = self.verifying.remove(&id.0) {
            if let (Some(key), Ok(output)) = (key, &output)
                && kept != *output
            {
                self.orders.push(JobOrder::Mismatch { key });
            }
            return;
        }
        if let (Some(key), Ok(output)) = (key, &output) {
            self.orders.push(JobOrder::Keep {
                key,
                output: output.clone(),
            });
        }
        finished.push((id, output));
    }

    /// Job `id` failed on its worker, for `error`: the engine records the
    /// failure, unless the job only verified a record the engine took.
    pub(crate) fn failed(&mut self, id: JobId, error: String, finished: &mut Vec<Finished>) {
        let Some(worker) = self.running.remove(&id.0) else {
            return;
        };
        self.workers[worker as usize].jobs.remove(&id.0);
        self.made(id, Err(error), finished);
    }

    /// Worker `worker` broke, for `error`: its jobs fail back to the engine
    /// and its tasks are gone; the host replaced it, or it takes nothing
    /// more.
    pub(crate) fn lost(
        &mut self,
        worker: u32,
        error: &str,
        replaced: bool,
        finished: &mut Vec<Finished>,
    ) {
        let Some(load) = self.workers.get_mut(worker as usize) else {
            return;
        };
        let jobs: Vec<u64> = load.jobs.drain().map(|(id, _)| id).collect();
        load.tasks = 0;
        load.lost = !replaced;
        for id in jobs {
            self.running.remove(&id);
            self.keys.remove(&id);
            if self.verifying.remove(&id).is_none() {
                finished.push((JobId(id), Err(error.to_owned())));
            }
        }
    }

    /// The worker a host's own task waits least on, counted on it until
    /// [`Self::task_done`].
    pub(crate) fn task(&mut self) -> Option<u32> {
        let worker = self.least()?;
        self.workers[worker as usize].tasks += 1;
        Some(worker)
    }

    /// Jobs it holds: on workers, in lookups, or to run in place.
    pub(crate) fn held(&self) -> u32 {
        (self.running.len() + self.looking.len() + self.here.len()) as u32
    }

    /// A host's task on `worker` ended.
    pub(crate) fn task_done(&mut self, worker: u32) {
        if let Some(load) = self.workers.get_mut(worker as usize) {
            load.tasks = load.tasks.saturating_sub(1);
        }
    }
}
