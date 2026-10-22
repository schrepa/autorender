//! The queue's submission side, where the work frames hand over is
//! finished and submitted, in the order it was handed over. Finishing an
//! encoder is where wgpu validates the passes it recorded and encodes them
//! for the GPU: most of what submitting costs.
//!
//! Natively a few threads of the renderer's own finish handovers side by
//! side while the frame records its next passes, and each submits its
//! handover in its turn. In browsers, whose renderer has one thread, a
//! handover is finished and submitted in place as it is handed over. Both
//! take the same steps in the same order: [`Handover::finish`], then the
//! queue's one submission.
//!
//! A native submission runs the map callbacks of the work the GPU finished
//! on the thread submitting it. Those of the renderer's maps wait for its
//! own thread (`platform::run_arrived_maps`), which runs them at each
//! handover and as it settles, where its own submissions ran them.

#[cfg(not(target_arch = "wasm32"))]
use std::{
    cell::Cell,
    sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, mpsc},
};

/// Recorded work a frame hands over: an encoder the submission side
/// finishes, or a command buffer finished already.
pub(crate) enum Recorded {
    Encoder(wgpu::CommandEncoder),
    Buffer(wgpu::CommandBuffer),
}

/// Recorded work handed over as one submission, in submission order.
#[derive(Default)]
pub(crate) struct Handover(Vec<Recorded>);

impl Handover {
    /// Adds `recorded` after the work the handover holds.
    pub(crate) fn push(&mut self, recorded: Recorded) {
        self.0.push(recorded);
    }

    /// Whether it holds no work.
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Its command buffers, in submission order, each encoder finished.
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

/// The queue's submission side.
pub(super) struct Submission {
    /// Natively, the threads finishing and submitting handovers: none in
    /// browsers, nor where none could start, where handovers finish and
    /// submit in place.
    #[cfg(not(target_arch = "wasm32"))]
    threads: Option<Threads>,
}

impl Submission {
    /// The submission side of `queue`: natively, its threads start.
    pub(super) fn new(queue: &wgpu::Queue) -> Self {
        #[cfg(target_arch = "wasm32")]
        let _ = queue;
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            threads: Threads::start(queue),
        }
    }

    /// Hands `handover` over, to be finished and submitted to `queue` after
    /// every handover before it: on the threads while the caller goes on,
    /// or in place, now.
    pub(super) fn hand(&self, queue: &wgpu::Queue, handover: Handover) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(threads) = &self.threads {
            threads.hand(handover);
            return;
        }
        super::submit(queue, handover.finish());
    }

    /// Waits until every handover so far is submitted.
    pub(super) fn settle(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(threads) = &self.threads {
            threads.settle();
        }
    }
}

/// Threads finishing handovers side by side: as many as the parts a frame
/// hands over, its prelude and its batches, where the processors spare
/// them beside the renderer's own.
#[cfg(not(target_arch = "wasm32"))]
const THREADS: usize = 4;

/// The submission threads, the handovers they take and whose turn it is to
/// submit.
#[cfg(not(target_arch = "wasm32"))]
struct Threads {
    /// The handovers, numbered in the order they were handed over, for
    /// whichever thread is free to take the next; closed as the threads
    /// drop.
    inbox: Option<mpsc::Sender<(u64, Handover)>>,
    workers: Vec<std::thread::JoinHandle<()>>,
    /// The handovers handed over so far: the next one's number.
    handed: Cell<u64>,
    /// The handovers handed over when the renderer last settled, all
    /// submitted since.
    settled: Cell<u64>,
    turns: Arc<Turns>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Threads {
    /// Starts the threads submitting to `queue`; none where none could
    /// start.
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

    /// Hands `handover` to whichever thread takes it, numbered after every
    /// handover before it; then runs the map callbacks the threads'
    /// submissions completed since.
    fn hand(&self, handover: Handover) {
        let number = self.handed.get();
        self.handed.set(number + 1);
        if let Some(inbox) = &self.inbox {
            // The threads take handovers until the inbox closes.
            let _ = inbox.send((number, handover));
        }
        crate::platform::run_arrived_maps();
    }

    /// Waits until every handover so far is submitted, then runs the map
    /// callbacks the threads' submissions completed.
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

/// A submission thread: takes the next handover, finishes it, and submits
/// it in its turn, until the inbox drops.
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

/// How many handovers are submitted, which says whose turn it is to
/// submit.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct Turns {
    submitted: Mutex<u64>,
    changed: Condvar,
}

#[cfg(not(target_arch = "wasm32"))]
impl Turns {
    /// Runs `submit`, the `number`th handover's submission, once every
    /// handover before it is submitted; the next then has its turn.
    fn take(&self, number: u64, submit: impl FnOnce()) {
        let mut submitted = self.reached(number);
        submit();
        *submitted = number + 1;
        self.changed.notify_all();
    }

    /// The count of handovers submitted, held, once it reaches `count`.
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
    /// Closes the inbox, so the threads stop once they have submitted every
    /// handover, and joins them.
    fn drop(&mut self) {
        self.inbox = None;
        for worker in self.workers.drain(..) {
            // Only a panic, which stops the engine, ends a thread early.
            let _ = worker.join();
        }
    }
}
