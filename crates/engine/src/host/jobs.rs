use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use engine_core::collections::HashMap;
use engine_core::{JobId, JobMessage, JobRef, JobRequest, JobShare, MISSING_RECORD, RecordKey};

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
        output: Arc<[u8]>,
    },
    Mismatch {
        key: String,
        kind: Option<JobRef>,
    },
}

pub(crate) type Finished = (JobId, Result<Arc<[u8]>, String>);

#[derive(Debug, Default)]
struct Load {
    tasks: u32,
    lost: bool,
}

#[derive(Debug, Default)]
struct Made {
    keep: Option<(RecordKey, JobRef)>,
    verifying: Option<Arc<[u8]>>,
}

#[derive(Debug)]
enum Job {
    Looking(JobRequest),
    Fetching,
    Waiting(JobRequest, Made),
    Here(Made),
    Running {
        worker: u32,
        priority: u32,
        made: Made,
    },
}

#[derive(Debug, Default)]
pub struct JobScheduler {
    workers: Vec<Load>,
    caching: bool,
    verify: bool,
    sample: Option<BTreeSet<String>>,
    jobs: BTreeMap<u64, Job>,
    fetched: HashMap<String, Arc<[u8]>>,
    orders: Vec<JobOrder>,
}

impl JobScheduler {
    pub(crate) fn set_workers(&mut self, workers: u32) -> bool {
        let busy = self.jobs.values().any(|job| match job {
            Job::Running { worker, .. } => *worker >= workers,
            Job::Looking(_) | Job::Fetching | Job::Waiting(..) | Job::Here(_) => false,
        });
        if !busy {
            self.workers.resize_with(workers as usize, Load::default);
        }
        !busy
    }

    pub(crate) fn working(&self) -> u32 {
        self.workers.iter().filter(|load| !load.lost).count() as u32
    }

    pub(crate) fn set_cache(&mut self, caching: bool, verify: bool) {
        self.caching = caching;
        self.verify = caching && verify;
    }

    pub(crate) fn set_sample(&mut self, sample: BTreeSet<String>) {
        self.sample = Some(sample);
    }

    fn verifies(&self, key: &str) -> bool {
        self.verify
            && self
                .sample
                .as_ref()
                .is_none_or(|sample| sample.contains(key))
    }

    pub(crate) fn take(
        &mut self,
        messages: Vec<JobMessage>,
        finished: &mut Vec<Finished>,
    ) -> Vec<JobOrder> {
        for message in messages {
            match message {
                JobMessage::Cancel(id) => {
                    if let Some(Job::Running { worker, .. }) = self.jobs.remove(&id.0) {
                        self.orders.push(JobOrder::Cancel { worker, id });
                    }
                }
                JobMessage::Share(JobShare::Clear) => {
                    self.jobs.clear();
                    self.fetched.clear();
                    self.orders.push(JobOrder::Share(JobShare::Clear));
                }
                JobMessage::Share(share) => self.orders.push(JobOrder::Share(share)),
                JobMessage::Run(request) => match request.record {
                    Some(record) if self.caching => {
                        let kind = Some(request.kind.clone());
                        self.lookup(request.id, record, kind);
                        self.jobs.insert(request.id.0, Job::Looking(request));
                    }
                    _ => self.dispatch(request, Made::default()),
                },
                JobMessage::Fetch(fetch) if self.caching => {
                    self.lookup(fetch.id, fetch.record, fetch.kind);
                    self.jobs.insert(fetch.id.0, Job::Fetching);
                }
                JobMessage::Fetch(fetch) => {
                    finished.push((fetch.id, Err(MISSING_RECORD.to_owned())));
                }
                JobMessage::Keep(record, kind, output, parity) => {
                    let key = record.hex();
                    if let Some(handed) = self.fetched.remove(&key) {
                        if !parity.holds(&handed, &output) {
                            self.orders.push(JobOrder::Mismatch { key, kind });
                        }
                    } else if self.caching {
                        let output = Arc::from(output);
                        self.orders.push(JobOrder::Keep { key, kind, output });
                    }
                }
            }
        }
        std::mem::take(&mut self.orders)
    }

    fn lookup(&mut self, id: JobId, record: RecordKey, kind: Option<JobRef>) {
        self.orders.push(JobOrder::Lookup {
            id,
            key: record.hex(),
            kind,
        });
    }

    pub(crate) fn kept(
        &mut self,
        id: JobId,
        key: String,
        output: Option<Arc<[u8]>>,
        finished: &mut Vec<Finished>,
    ) {
        let request = match self.jobs.remove(&id.0) {
            Some(Job::Fetching) => {
                if let (true, Some(output)) = (self.verifies(&key), &output) {
                    self.fetched.insert(key, output.clone());
                }
                finished.push((id, output.ok_or_else(|| MISSING_RECORD.to_owned())));
                return;
            }
            Some(Job::Looking(request)) => request,
            Some(job @ (Job::Waiting(..) | Job::Here(_) | Job::Running { .. })) => {
                self.jobs.insert(id.0, job);
                return;
            }
            None => return,
        };
        let made = match output {
            Some(output) if self.verifies(&key) => {
                finished.push((id, Ok(output.clone())));
                Made {
                    keep: request.record.map(|record| (record, request.kind.clone())),
                    verifying: Some(output),
                }
            }
            Some(output) => {
                finished.push((id, Ok(output)));
                return;
            }
            None => Made {
                keep: request.record.map(|record| (record, request.kind.clone())),
                verifying: None,
            },
        };
        self.dispatch(request, made);
    }

    fn dispatch(&mut self, request: JobRequest, made: Made) {
        let id = request.id.0;
        let Some(worker) = self.least() else {
            self.jobs.insert(id, Job::Waiting(request, made));
            return;
        };
        self.jobs.insert(
            id,
            Job::Running {
                worker,
                priority: request.priority,
                made,
            },
        );
        self.orders.push(JobOrder::Run { worker, request });
    }

    pub(crate) fn take_here(&mut self) -> Vec<JobRequest> {
        let waiting: Vec<u64> = self
            .jobs
            .iter()
            .filter(|(_, job)| matches!(job, Job::Waiting(..)))
            .map(|(&id, _)| id)
            .collect();
        let mut here = Vec::with_capacity(waiting.len());
        for id in waiting {
            if let Some(Job::Waiting(request, made)) = self.jobs.remove(&id) {
                self.jobs.insert(id, Job::Here(made));
                here.push(request);
            }
        }
        here
    }

    fn least(&self) -> Option<u32> {
        self.workers
            .iter()
            .enumerate()
            .filter(|(_, load)| !load.lost)
            .min_by_key(|&(worker, load)| {
                let (count, urgent) = self.running_on(worker as u32);
                (count + load.tasks as usize, std::cmp::Reverse(urgent))
            })
            .map(|(worker, _)| worker as u32)
    }

    fn running_on(&self, on: u32) -> (usize, u32) {
        self.jobs
            .values()
            .filter_map(|job| match job {
                Job::Running {
                    worker, priority, ..
                } if *worker == on => Some(*priority),
                Job::Running { .. }
                | Job::Looking(_)
                | Job::Fetching
                | Job::Waiting(..)
                | Job::Here(_) => None,
            })
            .fold((0, u32::MAX), |(count, urgent), priority| {
                (count + 1, urgent.min(priority))
            })
    }

    pub(crate) fn done(&mut self, id: JobId, output: Arc<[u8]>, finished: &mut Vec<Finished>) {
        if let Some(made) = self.ended(id) {
            self.made(id, made, Ok(output), finished);
        }
    }

    pub(crate) fn ran_here(
        &mut self,
        id: JobId,
        output: Result<Vec<u8>, String>,
        finished: &mut Vec<Finished>,
    ) {
        match self.jobs.remove(&id.0) {
            Some(Job::Here(made)) => self.made(id, made, output.map(Arc::from), finished),
            Some(job) => {
                self.jobs.insert(id.0, job);
            }
            None => {}
        }
    }

    fn ended(&mut self, id: JobId) -> Option<Made> {
        match self.jobs.remove(&id.0)? {
            Job::Running { made, .. } => Some(made),
            job @ (Job::Looking(_) | Job::Fetching | Job::Waiting(..) | Job::Here(_)) => {
                self.jobs.insert(id.0, job);
                None
            }
        }
    }

    fn made(
        &mut self,
        id: JobId,
        made: Made,
        output: Result<Arc<[u8]>, String>,
        finished: &mut Vec<Finished>,
    ) {
        let Made { keep, verifying } = made;
        if let Some(kept) = verifying {
            if let (Some((key, kind)), Ok(output)) = (keep, &output)
                && kept != *output
            {
                self.orders.push(JobOrder::Mismatch {
                    key: key.hex(),
                    kind: Some(kind),
                });
            }
            return;
        }
        if let (Some((key, kind)), Ok(output)) = (keep, &output) {
            self.orders.push(JobOrder::Keep {
                key: key.hex(),
                kind: Some(kind),
                output: output.clone(),
            });
        }
        finished.push((id, output));
    }

    pub(crate) fn failed(&mut self, id: JobId, error: String, finished: &mut Vec<Finished>) {
        if let Some(made) = self.ended(id) {
            self.made(id, made, Err(error), finished);
        }
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
        load.tasks = 0;
        load.lost = !replaced;
        let gone: Vec<u64> = self
            .jobs
            .iter()
            .filter(|(_, job)| matches!(job, Job::Running { worker: on, .. } if *on == worker))
            .map(|(&id, _)| id)
            .collect();
        for id in gone {
            if let Some(Job::Running { made, .. }) = self.jobs.remove(&id)
                && made.verifying.is_none()
            {
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
        self.jobs.len() as u32
    }

    pub(crate) fn task_done(&mut self, worker: u32) {
        if let Some(load) = self.workers.get_mut(worker as usize) {
            load.tasks = load.tasks.saturating_sub(1);
        }
    }
}
