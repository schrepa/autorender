#[cfg(not(target_arch = "wasm32"))]
use std::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, mpsc};

use super::Compiled;

pub(crate) enum Created {
    Render(wgpu::RenderPipeline),
    Compute(wgpu::ComputePipeline),
}

pub(crate) struct Finished {
    pub(crate) program: u64,
    pub(crate) key: u64,
    pub(crate) pass: &'static str,
    pub(crate) noted: Option<(Compiled, bool)>,
    pub(crate) made: Result<Created, wgpu::Error>,
}

#[cfg(not(target_arch = "wasm32"))]
type Job = Box<dyn FnOnce() -> Finished + Send>;

#[derive(Default)]
pub(crate) struct Making {
    #[cfg(not(target_arch = "wasm32"))]
    started: bool,
    #[cfg(not(target_arch = "wasm32"))]
    threads: Option<Threads>,
    inline: Vec<Finished>,
    pending: Vec<(u64, u64)>,
    refused: Vec<(u64, u64)>,
}

impl Making {
    pub(crate) fn pending(&self) -> u32 {
        u32::try_from(self.pending.len()).unwrap_or(u32::MAX)
    }

    pub(crate) fn first(&self) -> Option<(u64, u64)> {
        self.pending.first().copied()
    }

    pub(crate) fn waits(&self, made: (u64, u64)) -> bool {
        self.pending.contains(&made)
    }

    pub(crate) fn refused(&self, made: (u64, u64)) -> bool {
        self.refused.contains(&made)
    }

    pub(crate) fn refuse(&mut self, made: (u64, u64)) {
        self.refused.push(made);
    }

    pub(crate) fn forget_refused(&mut self) {
        self.refused.clear();
    }

    pub(crate) fn submit(
        &mut self,
        made: (u64, u64),
        job: impl FnOnce() -> Finished + wgpu::WasmNotSend + 'static,
    ) {
        self.pending.push(made);
        #[cfg(not(target_arch = "wasm32"))]
        {
            if !self.started {
                self.started = true;
                self.threads = Threads::start();
            }
            if let Some(threads) = &self.threads {
                threads.hand(made, Box::new(job));
                return;
            }
        }
        self.inline.push(job());
    }

    pub(crate) fn take(&mut self, waited: Option<(u64, u64)>) -> Vec<Finished> {
        let mut taken = std::mem::take(&mut self.inline);
        let arrived = self.arrive(&taken, waited);
        taken.extend(arrived);
        self.pending.retain(|made| {
            !taken
                .iter()
                .any(|finished| (finished.program, finished.key) == *made)
        });
        taken
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn arrive(&self, taken: &[Finished], waited: Option<(u64, u64)>) -> Vec<Finished> {
        let mut arrived = Vec::new();
        let Some(threads) = &self.threads else {
            return arrived;
        };
        let mut waiting = waited.filter(|&made| {
            self.pending.contains(&made)
                && !taken
                    .iter()
                    .any(|finished| (finished.program, finished.key) == made)
        });
        if let Some(job) = waiting.and_then(|made| threads.claim(made)) {
            arrived.push(job());
            waiting = None;
        }
        loop {
            let next = if waiting.is_some() {
                threads.finished.recv().ok()
            } else {
                threads.finished.try_recv().ok()
            };
            let Some(finished) = next else {
                break;
            };
            if waiting == Some((finished.program, finished.key)) {
                waiting = None;
            }
            arrived.push(finished);
        }
        arrived
    }

    #[cfg(target_arch = "wasm32")]
    fn arrive(&self, _: &[Finished], _: Option<(u64, u64)>) -> Vec<Finished> {
        Vec::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct Queue {
    jobs: VecDeque<((u64, u64), Job)>,
    stopped: bool,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct Shared {
    queue: Mutex<Queue>,
    handed: Condvar,
}

#[cfg(not(target_arch = "wasm32"))]
impl Shared {
    fn lock(&self) -> MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct Threads {
    shared: Arc<Shared>,
    finished: mpsc::Receiver<Finished>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Threads {
    fn start() -> Option<Self> {
        let count = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .saturating_sub(1)
            .max(1);
        let shared = Arc::new(Shared::default());
        let (done, finished) = mpsc::channel::<Finished>();
        let workers: Vec<_> = (0..count)
            .map_while(|index| {
                let (shared, done) = (Arc::clone(&shared), done.clone());
                std::thread::Builder::new()
                    .name(format!("engine-pipelines-{index}"))
                    .spawn(move || work(&shared, &done))
                    .ok()
            })
            .collect();
        if workers.is_empty() {
            return None;
        }
        Some(Self {
            shared,
            finished,
            workers,
        })
    }

    fn hand(&self, made: (u64, u64), job: Job) {
        self.shared.lock().jobs.push_back((made, job));
        self.shared.handed.notify_one();
    }

    fn claim(&self, made: (u64, u64)) -> Option<Job> {
        let mut queue = self.shared.lock();
        let at = queue.jobs.iter().position(|(known, _)| *known == made)?;
        queue.jobs.remove(at).map(|(_, job)| job)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn work(shared: &Shared, done: &mpsc::Sender<Finished>) {
    loop {
        let job = {
            let mut queue = shared.lock();
            loop {
                if queue.stopped {
                    return;
                }
                if let Some((_, job)) = queue.jobs.pop_front() {
                    break job;
                }
                queue = shared
                    .handed
                    .wait(queue)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        };
        if done.send(job()).is_err() {
            return;
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for Threads {
    fn drop(&mut self) {
        {
            let mut queue = self.shared.lock();
            queue.stopped = true;
            queue.jobs.clear();
        }
        self.shared.handed.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
