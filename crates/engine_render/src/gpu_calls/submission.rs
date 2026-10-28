#[cfg(not(target_arch = "wasm32"))]
use std::{
    cell::Cell,
    sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, mpsc},
};

pub(crate) enum Recorded {
    Encoder(wgpu::CommandEncoder),
    Buffer(wgpu::CommandBuffer),
}

#[derive(Default)]
pub(crate) struct Handover(Vec<Recorded>);

impl Handover {
    pub(crate) fn push(&mut self, recorded: Recorded) {
        self.0.push(recorded);
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn finish(self) -> Vec<wgpu::CommandBuffer> {
        self.0
            .into_iter()
            .map(|recorded| match recorded {
                Recorded::Encoder(encoder) => encoder.finish(),
                Recorded::Buffer(buffer) => buffer,
            })
            .collect()
    }
}

impl From<wgpu::CommandEncoder> for Handover {
    fn from(encoder: wgpu::CommandEncoder) -> Self {
        Self(vec![Recorded::Encoder(encoder)])
    }
}

pub(super) struct Submission {
    #[cfg(not(target_arch = "wasm32"))]
    threads: Option<Threads>,
}

impl Submission {
    pub(super) fn new(queue: &wgpu::Queue) -> Self {
        #[cfg(target_arch = "wasm32")]
        let _ = queue;
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            threads: Threads::start(queue),
        }
    }

    pub(super) fn hand(&self, queue: &wgpu::Queue, handover: Handover) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(threads) = &self.threads {
            threads.hand(handover);
            return;
        }
        super::submit(queue, handover.finish());
    }

    pub(super) fn settle(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(threads) = &self.threads {
            threads.settle();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
const THREADS: usize = 4;

#[cfg(not(target_arch = "wasm32"))]
struct Threads {
    inbox: Option<mpsc::Sender<(u64, Handover)>>,
    workers: Vec<std::thread::JoinHandle<()>>,
    handed: Cell<u64>,
    settled: Cell<u64>,
    turns: Arc<Turns>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Threads {
    fn start(queue: &wgpu::Queue) -> Option<Self> {
        let spare = std::thread::available_parallelism().map_or(1, usize::from) - 1;
        let (inbox, handovers) = mpsc::channel::<(u64, Handover)>();
        let handovers = Arc::new(Mutex::new(handovers));
        let turns = Arc::new(Turns::default());
        let workers: Vec<_> = (0..spare.clamp(1, THREADS))
            .map_while(|index| {
                let (queue, handovers, turns) =
                    (queue.clone(), Arc::clone(&handovers), Arc::clone(&turns));
                std::thread::Builder::new()
                    .name(format!("engine-submit-{index}"))
                    .spawn(move || work(&queue, &handovers, &turns))
                    .ok()
            })
            .collect();
        if workers.is_empty() {
            return None;
        }
        Some(Self {
            inbox: Some(inbox),
            workers,
            handed: Cell::new(0),
            settled: Cell::new(0),
            turns,
        })
    }

    fn hand(&self, handover: Handover) {
        let number = self.handed.get();
        self.handed.set(number + 1);
        if let Some(inbox) = &self.inbox {
            let _ = inbox.send((number, handover));
        }
        crate::platform::run_arrived_maps();
    }

    fn settle(&self) {
        let handed = self.handed.get();
        if self.settled.replace(handed) == handed {
            return;
        }
        let _profile = engine_profile::span("render", "submission_wait");
        drop(self.turns.reached(handed));
        crate::platform::run_arrived_maps();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn work(queue: &wgpu::Queue, handovers: &Mutex<mpsc::Receiver<(u64, Handover)>>, turns: &Turns) {
    loop {
        let next = handovers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .recv();
        let Ok((number, handover)) = next else {
            return;
        };
        let buffers = handover.finish();
        turns.take(number, || {
            super::submit(queue, buffers);
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct Turns {
    submitted: Mutex<u64>,
    changed: Condvar,
}

#[cfg(not(target_arch = "wasm32"))]
impl Turns {
    fn take(&self, number: u64, submit: impl FnOnce()) {
        let mut submitted = self.reached(number);
        submit();
        *submitted = number + 1;
        self.changed.notify_all();
    }

    fn reached(&self, count: u64) -> MutexGuard<'_, u64> {
        let submitted = self
            .submitted
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.changed
            .wait_while(submitted, |submitted| *submitted < count)
            .unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for Threads {
    fn drop(&mut self) {
        self.inbox = None;
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
