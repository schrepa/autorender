use std::sync::Arc;

use engine_core::collections::HashMap;
use engine_core::{JobRef, JobStats, RecordKey};

pub const MISSING_RECORD: &str = "no store holds the record";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(pub u64);

#[derive(Clone, Debug, PartialEq)]
pub struct JobRequest {
    pub id: JobId,
    pub kind: JobRef,
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

#[derive(Clone, Debug, PartialEq)]
pub enum JobShare {
    Clear,
    Set { key: RecordKey, bytes: Arc<[u8]> },
    Remove { key: RecordKey },
}

#[derive(Debug)]
pub enum JobMessage {
    Cancel(JobId),
    Share(JobShare),
    Run(JobRequest),
    Fetch(RecordFetch),
    Keep(RecordKey, Option<JobRef>, Vec<u8>),
}

#[derive(Debug)]
enum Asker {
    World(RecordKey),
    Host,
}

#[derive(Debug, Default)]
pub(crate) struct HostJobs {
    next: u64,
    asked: HashMap<u64, Asker>,
    running: HashMap<[u8; 32], JobId>,
    held: HashMap<u64, Result<Vec<u8>, String>>,
    outbox: Vec<JobMessage>,
    shared: HashMap<[u8; 32], u64>,
    stats: JobStats,
}

impl HostJobs {
    fn id(&mut self) -> JobId {
        self.next += 1;
        JobId(self.next)
    }

    pub(crate) fn take(&mut self, orders: Vec<engine_registry::JobOrder>) -> Vec<JobMessage> {
        use engine_registry::JobOrder as World;
        for order in orders {
            match order {
                World::Start {
                    key,
                    kind,
                    source,
                    input,
                    priority,
                } => self.run(key, JobRef::named(kind, source), input, priority, false),
                World::Fetch {
                    key,
                    kind,
                    source,
                    input,
                    priority,
                } => self.run(key, JobRef::named(kind, source), input, priority, true),
                World::Cancel { key } => {
                    if let Some(id) = self.running.remove(&key) {
                        self.asked.remove(&id.0);
                        self.stats.cancelled += 1;
                        self.outbox.push(JobMessage::Cancel(id));
                    }
                }
                World::Share { key, bytes } => {
                    self.shared.insert(key, bytes.len() as u64);
                    self.outbox.push(JobMessage::Share(JobShare::Set {
                        key: RecordKey::from_hash(key),
                        bytes,
                    }));
                }
                World::Keep {
                    key,
                    kind,
                    source,
                    bytes,
                } => self.outbox.push(JobMessage::Keep(
                    RecordKey::from_hash(key),
                    Some(JobRef::named(kind, source)),
                    bytes,
                )),
            }
        }
        std::mem::take(&mut self.outbox)
    }

    fn run(&mut self, key: [u8; 32], kind: JobRef, input: Vec<u8>, priority: u32, cooked: bool) {
        if self.running.contains_key(&key) {
            return;
        }
        let id = self.id();
        let record = RecordKey::from_hash(key);
        self.running.insert(key, id);
        self.asked.insert(id.0, Asker::World(record));
        self.stats.submitted += 1;
        self.outbox.push(JobMessage::Run(JobRequest {
            id,
            kind,
            input,
            priority,
            record: cooked.then_some(record),
        }));
    }

    pub(crate) fn fetch_here(&mut self, record: RecordKey, priority: u32) -> JobId {
        let id = self.id();
        self.asked.insert(id.0, Asker::Host);
        self.outbox
            .push(JobMessage::Fetch(RecordFetch { id, record, priority }));
        id
    }

    pub(crate) fn keep_here(&mut self, record: RecordKey, bytes: Vec<u8>) {
        self.outbox.push(JobMessage::Keep(record, None, bytes));
    }

    pub(crate) fn take_here(&mut self, id: JobId) -> Option<Result<Vec<u8>, String>> {
        self.held.remove(&id.0)
    }

    pub(crate) fn finish(
        &mut self,
        id: JobId,
        output: Result<Vec<u8>, String>,
    ) -> Option<(RecordKey, Result<Vec<u8>, String>)> {
        match self.asked.remove(&id.0)? {
            Asker::World(key) => {
                self.running.remove(&key.hash());
                match &output {
                    Ok(_) => self.stats.completed += 1,
                    Err(_) => self.stats.failed += 1,
                }
                Some((key, output))
            }
            Asker::Host => {
                self.held.insert(id.0, output);
                None
            }
        }
    }

    pub(crate) fn clear(&mut self) {
        self.asked.clear();
        self.running.clear();
        self.held.clear();
        self.shared.clear();
        self.outbox.clear();
        self.outbox.push(JobMessage::Share(JobShare::Clear));
    }

    pub(crate) fn set_workers(&mut self, workers: u32) {
        self.stats.workers = workers;
    }

    pub(crate) fn workers(&self) -> u32 {
        self.stats.workers
    }

    pub(crate) fn ran_here(&mut self) {
        self.stats.in_place += 1;
    }

    pub(crate) fn report_cache(&mut self, hits: u64, misses: u64, bytes: u64, mismatches: u64) {
        self.stats.cache_hits = hits;
        self.stats.cache_misses = misses;
        self.stats.cache_bytes = bytes;
        self.stats.cache_mismatches = mismatches;
    }

    pub(crate) fn report_held(&mut self, held: u32) {
        self.stats.held = held;
    }

    pub(crate) fn stats(&self) -> JobStats {
        JobStats {
            running: self.running.len() as u32,
            shares: self.shared.len() as u32,
            shared_bytes: self.shared.values().sum(),
            ..self.stats
        }
    }
}

#[derive(Debug)]
pub enum JobOrder {
    Share(JobShare),
    Run {
        worker: u32,
        request: JobRequest,
    },
    Cancel {
        worker: u32,
        id: JobId,
    },
    Lookup {
        id: JobId,
        key: String,
        kind: Option<JobRef>,
    },
    Keep {
        key: String,
        kind: Option<JobRef>,
        output: Vec<u8>,
    },
    Mismatch {
        key: String,
        kind: Option<JobRef>,
    },
}

pub(crate) type Finished = (JobId, Result<Vec<u8>, String>);

#[derive(Debug, Default)]
struct Load {
    jobs: HashMap<u64, u32>,
    tasks: u32,
    lost: bool,
}

#[derive(Debug)]
enum Looking {
    Job(JobRequest),
    Fetch,
}

#[derive(Debug, Default)]
pub struct JobScheduler {
    workers: Vec<Load>,
    caching: bool,
    verify: bool,
    running: HashMap<u64, u32>,
    keys: HashMap<u64, (RecordKey, JobRef)>,
    verifying: HashMap<u64, Vec<u8>>,
    fetched: HashMap<String, Vec<u8>>,
    looking: HashMap<u64, Looking>,
    here: Vec<JobRequest>,
    orders: Vec<JobOrder>,
}

impl JobScheduler {
    pub(crate) fn set_workers(&mut self, workers: u32) {
        self.workers.resize_with(workers as usize, Load::default);
    }

    pub(crate) fn working(&self) -> u32 {
        self.workers.iter().filter(|load| !load.lost).count() as u32
    }

    pub(crate) fn set_cache(&mut self, caching: bool, verify: bool) {
        self.caching = caching;
        self.verify = caching && verify;
    }

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
                    self.fetched.clear();
                    self.looking.clear();
                    self.here.clear();
                    self.orders.push(JobOrder::Share(JobShare::Clear));
                }
                JobMessage::Share(share) => self.orders.push(JobOrder::Share(share)),
                JobMessage::Run(request) => match request.record {
                    Some(record) if self.caching => {
                        let kind = Some(request.kind.clone());
                        self.lookup(request.id, record, kind, Looking::Job(request));
                    }
                    _ => self.dispatch(request),
                },
                JobMessage::Fetch(fetch) if self.caching => {
                    self.lookup(fetch.id, fetch.record, None, Looking::Fetch);
                }
                JobMessage::Fetch(fetch) => {
                    finished.push((fetch.id, Err(MISSING_RECORD.to_owned())));
                }
                JobMessage::Keep(record, kind, output) => {
                    let key = record.hex();
                    if let Some(handed) = self.fetched.remove(&key) {
                        if handed != output {
                            self.orders.push(JobOrder::Mismatch { key, kind });
                        }
                    } else if self.caching {
                        self.orders.push(JobOrder::Keep { key, kind, output });
                    }
                }
            }
        }
        std::mem::take(&mut self.orders)
    }

    fn lookup(&mut self, id: JobId, record: RecordKey, kind: Option<JobRef>, looking: Looking) {
        self.orders.push(JobOrder::Lookup {
            id,
            key: record.hex(),
            kind,
        });
        self.looking.insert(id.0, looking);
    }

    pub(crate) fn kept(
        &mut self,
        id: JobId,
        key: String,
        output: Option<Vec<u8>>,
        finished: &mut Vec<Finished>,
    ) {
        let Some(looking) = self.looking.remove(&id.0) else {
            return;
        };
        let request = match looking {
            Looking::Fetch => {
                if let (true, Some(output)) = (self.verify, &output) {
                    self.fetched.insert(key, output.clone());
                }
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
        if let Some(record) = request.record {
            self.keys.insert(id.0, (record, request.kind.clone()));
        }
        self.dispatch(request);
    }

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

    pub(crate) fn take_here(&mut self) -> Vec<JobRequest> {
        std::mem::take(&mut self.here)
    }

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

    pub(crate) fn done(&mut self, id: JobId, output: Vec<u8>, finished: &mut Vec<Finished>) {
        let Some(worker) = self.running.remove(&id.0) else {
            return;
        };
        self.workers[worker as usize].jobs.remove(&id.0);
        self.made(id, Ok(output), finished);
    }

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
            if let (Some((key, kind)), Ok(output)) = (key, &output)
                && kept != *output
            {
                self.orders.push(JobOrder::Mismatch {
                    key: key.hex(),
                    kind: Some(kind),
                });
            }
            return;
        }
        if let (Some((key, kind)), Ok(output)) = (key, &output) {
            self.orders.push(JobOrder::Keep {
                key: key.hex(),
                kind: Some(kind),
                output: output.clone(),
            });
        }
        finished.push((id, output));
    }

    pub(crate) fn failed(&mut self, id: JobId, error: String, finished: &mut Vec<Finished>) {
        let Some(worker) = self.running.remove(&id.0) else {
            return;
        };
        self.workers[worker as usize].jobs.remove(&id.0);
        self.made(id, Err(error), finished);
    }

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

    pub(crate) fn task(&mut self) -> Option<u32> {
        let worker = self.least()?;
        self.workers[worker as usize].tasks += 1;
        Some(worker)
    }

    pub(crate) fn held(&self) -> u32 {
        (self.running.len() + self.looking.len() + self.here.len()) as u32
    }

    pub(crate) fn task_done(&mut self, worker: u32) {
        if let Some(load) = self.workers.get_mut(worker as usize) {
            load.tasks = load.tasks.saturating_sub(1);
        }
    }
}
