use std::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, mpsc};
#[cfg(target_arch = "wasm32")]
use std::{cell::RefCell, future::Future, pin::Pin, rc::Rc};

pub(crate) enum Created {
    Render(wgpu::RenderPipeline),
    Compute(wgpu::ComputePipeline),
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) type Creating = Result<Created, wgpu::Error>;
#[cfg(target_arch = "wasm32")]
pub(crate) type Creating = Pin<Box<dyn Future<Output = Result<Created, wgpu::Error>>>>;

pub(crate) struct Finished {
    pub(crate) program: u64,
    pub(crate) key: u64,
    pub(crate) pass: &'static str,
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
    #[cfg(not(target_arch = "wasm32"))]
    inline: Vec<Finished>,
    #[cfg(target_arch = "wasm32")]
    flights: Option<Flights>,
    pending: Vec<(u64, u64)>,
    later: Vec<(u64, u64)>,
    refused: Vec<(u64, u64)>,
    pub(crate) book: crate::pipelines::Book,
}

impl Making {
    pub(crate) fn pending(&self) -> u32 {
        u32::try_from(self.pending.len()).unwrap_or(u32::MAX)
    }

    pub(crate) fn needed(&self) -> u32 {
        self.pending().saturating_sub(self.later_left())
    }

    pub(crate) fn later_left(&self) -> u32 {
        u32::try_from(self.later.len()).unwrap_or(u32::MAX)
    }

    pub(crate) fn first(&self) -> Option<(u64, u64)> {
        self.pending.first().copied()
    }

    pub(crate) fn waits(&self, made: (u64, u64)) -> bool {
        self.pending.contains(&made)
    }

    pub(crate) fn program_waits(&self, program: u64) -> bool {
        self.pending.iter().any(|(known, _)| *known == program)
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
        (made, later): ((u64, u64), bool),
        (pass, stages): (&'static str, Vec<u64>),
        job: impl FnOnce() -> Creating + wgpu::WasmNotSend + 'static,
    ) {
        self.pending.push(made);
        if later {
            self.later.push(made);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            drop(stages);
            let (program, key) = made;
            let job = move || Finished {
                program,
                key,
                pass,
                made: job(),
            };
            if !self.started {
                self.started = true;
                self.threads = Threads::start();
            }
            match &self.threads {
                Some(threads) => threads.hand((made, later), Box::new(job)),
                None => self.inline.push(job()),
            }
        }
        #[cfg(target_arch = "wasm32")]
        self.flights
            .get_or_insert_with(Flights::start)
            .hand(Queued {
                made,
                pass,
                later,
                stages,
                job: Box::new(job),
            });
    }

    pub(crate) fn take(&mut self, waited: Option<(u64, u64)>) -> Vec<Finished> {
        #[cfg(not(target_arch = "wasm32"))]
        let taken = {
            let mut taken = std::mem::take(&mut self.inline);
            let arrived = self.arrive(&taken, waited);
            taken.extend(arrived);
            taken
        };
        #[cfg(target_arch = "wasm32")]
        let taken = self
            .flights
            .as_ref()
            .map(|flights| flights.arrive(waited))
            .unwrap_or_default();
        self.pending.retain(|made| {
            !taken
                .iter()
                .any(|finished| (finished.program, finished.key) == *made)
        });
        let pending = &self.pending;
        self.later.retain(|made| pending.contains(made));
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
}

#[cfg(target_arch = "wasm32")]
struct Queued {
    made: (u64, u64),
    pass: &'static str,
    later: bool,
    stages: Vec<u64>,
    job: Box<dyn FnOnce() -> Creating>,
}

#[cfg(target_arch = "wasm32")]
struct Flight {
    jobs: VecDeque<Queued>,
    arrived: Vec<Finished>,
    flying: usize,
    slots: usize,
    compiling: Vec<u64>,
}

#[cfg(target_arch = "wasm32")]
impl Flight {
    fn next(&mut self) -> Option<Queued> {
        if self.flying >= self.slots {
            return None;
        }
        let needed = self.jobs.iter().any(|queued| !queued.later);
        let at = self.jobs.iter().position(|queued| {
            (!queued.later || !needed)
                && !queued
                    .stages
                    .iter()
                    .any(|stage| self.compiling.contains(stage))
        })?;
        let queued = self.jobs.remove(at)?;
        self.compiling.extend(queued.stages.iter().copied());
        self.flying += 1;
        Some(queued)
    }

    fn land(&mut self, stages: &[u64]) {
        for stage in stages {
            if let Some(at) = self.compiling.iter().position(|known| known == stage) {
                self.compiling.swap_remove(at);
            }
        }
        self.flying -= 1;
    }
}

#[cfg(target_arch = "wasm32")]
struct Flights {
    shared: Rc<RefCell<Flight>>,
}

#[cfg(target_arch = "wasm32")]
impl Flights {
    fn start() -> Self {
        let cores = js_sys::Reflect::get(&js_sys::global(), &"navigator".into())
            .and_then(|navigator| js_sys::Reflect::get(&navigator, &"hardwareConcurrency".into()))
            .ok()
            .and_then(|cores| cores.as_f64())
            .map_or(1, |cores| cores as usize);
        Self {
            shared: Rc::new(RefCell::new(Flight {
                jobs: VecDeque::new(),
                arrived: Vec::new(),
                flying: 0,
                slots: cores.saturating_sub(1).max(1),
                compiling: Vec::new(),
            })),
        }
    }

    fn hand(&self, queued: Queued) {
        {
            let jobs = &mut self.shared.borrow_mut().jobs;
            match jobs.iter().position(|held| held.later && !queued.later) {
                Some(at) => jobs.insert(at, queued),
                None => jobs.push_back(queued),
            }
        }
        launch(&self.shared);
    }

    fn arrive(&self, waited: Option<(u64, u64)>) -> Vec<Finished> {
        let mut flight = self.shared.borrow_mut();
        if let Some(at) =
            waited.and_then(|made| flight.jobs.iter().position(|queued| queued.made == made))
            && let Some(queued) = flight.jobs.remove(at)
        {
            flight.jobs.push_front(queued);
        }
        std::mem::take(&mut flight.arrived)
    }
}

#[cfg(target_arch = "wasm32")]
fn launch(shared: &Rc<RefCell<Flight>>) {
    loop {
        let next = shared.borrow_mut().next();
        let Some(queued) = next else {
            return;
        };
        wasm_bindgen_futures::spawn_local(fly(Rc::clone(shared), queued));
    }
}

#[cfg(target_arch = "wasm32")]
async fn fly(shared: Rc<RefCell<Flight>>, queued: Queued) {
    let Queued {
        made: (program, key),
        pass,
        stages,
        job,
        ..
    } = queued;
    let made = job().await;
    {
        let mut flight = shared.borrow_mut();
        flight.land(&stages);
        flight.arrived.push(Finished {
            program,
            key,
            pass,
            made,
        });
    }
    launch(&shared);
}

#[cfg(target_arch = "wasm32")]
impl Drop for Flights {
    fn drop(&mut self) {
        self.shared.borrow_mut().jobs.clear();
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct Queue {
    jobs: VecDeque<((u64, u64), bool, Job)>,
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
                    .spawn(move || {
                        engine_profile::announce_thread(format_args!("Pipelines {index}"));
                        work(&shared, &done);
                    })
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

    fn hand(&self, (made, later): ((u64, u64), bool), job: Job) {
        {
            let mut queue = self.shared.lock();
            match queue.jobs.iter().position(|held| held.1 && !later) {
                Some(at) => queue.jobs.insert(at, (made, later, job)),
                None => queue.jobs.push_back((made, later, job)),
            }
        }
        self.shared.handed.notify_one();
    }

    fn claim(&self, made: (u64, u64)) -> Option<Job> {
        let mut queue = self.shared.lock();
        let at = queue.jobs.iter().position(|held| held.0 == made)?;
        queue.jobs.remove(at).map(|(_, _, job)| job)
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
                if let Some((_, _, job)) = queue.jobs.pop_front() {
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
