use engine_core::collections::HashMap;
use engine_core::{JobId, JobRef, JobRequest, JobShare, MISSING_RECORD, RecordKey};

use crate::app_host::JobMessage;

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
                JobMessage::Keep(record, output, parity) => {
                    let key = record.hex();
                    if let Some(handed) = self.fetched.remove(&key) {
                        if !parity.holds(&handed, &output) {
                            self.orders.push(JobOrder::Mismatch { key, kind: None });
                        }
                    } else if self.caching {
                        self.orders.push(JobOrder::Keep {
                            key,
                            kind: None,
                            output,
                        });
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
