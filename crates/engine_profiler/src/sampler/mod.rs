//! CPU samples of chosen threads of this program, taken by a thread of its
//! own at a fixed interval: each sample is the thread's call stack, unwound
//! in process, so no privilege or setup is needed on any operating system.
//! Windows and macOS suspend the thread and read its registers; Linux
//! interrupts it with a signal. Windows unwinds with the system's unwind
//! tables; macOS and Linux follow frame pointers (Apple's ABIs keep them;
//! on Linux, builds keep them with `-C force-frame-pointers=yes`). Threads
//! register themselves, recording their stack's bounds, which no unwind
//! reads outside. Samples never allocate while a thread is stopped.

#![cfg_attr(
    any(unix, target_arch = "x86_64"),
    expect(
        unsafe_code,
        reason = "follows frame pointers within a thread's stack and reads sampled code"
    )
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[cfg(windows)]
#[path = "windows.rs"]
mod os;

#[cfg(target_vendor = "apple")]
#[path = "macos.rs"]
mod os;

#[cfg(all(unix, not(target_vendor = "apple")))]
#[path = "linux.rs"]
mod os;

#[cfg(not(any(windows, unix)))]
mod os {
    pub(super) const METHOD: Option<&str> = None;
    #[derive(Clone, Copy)]
    pub(super) struct Id;
    pub(super) struct Open;
    pub(super) fn current() -> Option<(Id, (u64, u64))> {
        None
    }
    pub(super) fn open(_: Id) -> Option<Open> {
        None
    }
    pub(super) fn sample(_: &Open, _: (u64, u64), _: &mut [u64]) -> usize {
        0
    }
    pub(super) fn cpu_ns(_: &Open) -> Option<u64> {
        None
    }
}

/// Frames a sample keeps at most, leaf first.
const MAX_FRAMES: usize = 256;

/// How this operating system's samples are taken; `None` where none are.
pub fn method() -> Option<&'static str> {
    os::METHOD
}

/// A thread to sample: registered by the thread itself.
#[derive(Clone)]
pub struct SampledThread {
    pub name: String,
    id: os::Id,
    // The stack's lowest and one-past-highest address.
    bounds: (u64, u64),
}

impl SampledThread {
    /// The calling thread, named `name`; `None` where threads cannot be
    /// sampled.
    pub fn current(name: impl Into<String>) -> Option<Self> {
        let (id, bounds) = os::current()?;
        Some(Self {
            name: name.into(),
            id,
            bounds,
        })
    }
}

/// One sample: when, the stack (instruction pointer then return
/// addresses), and the thread's CPU time since the previous sample.
pub struct Sample {
    pub time_ms: f64,
    pub stack: Vec<u64>,
    pub cpu_ns: Option<u64>,
}

/// A thread's samples, and the ticks that took none (the thread had
/// exited, or would not stop).
pub struct ThreadSamples {
    pub name: String,
    pub samples: Vec<Sample>,
    pub missed: u64,
}

/// What a sampler took.
pub struct Samples {
    pub threads: Vec<ThreadSamples>,
    pub interval_ms: f64,
    pub method: &'static str,
}

/// A sampler running.
pub struct Sampler {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<Samples>,
}

impl Sampler {
    /// Samples `threads` every `interval`, timing samples by `clock`
    /// (milliseconds); `None` where threads cannot be sampled.
    pub fn start(
        threads: Vec<SampledThread>,
        interval: Duration,
        clock: fn() -> f64,
    ) -> Option<Self> {
        let method = os::METHOD?;
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("engine-sampler".into())
            .spawn(move || run(&threads, interval, clock, &stopping, method))
            .ok()?;
        Some(Self { stop, thread })
    }

    /// Stops sampling: the samples taken.
    pub fn finish(self) -> Samples {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.join().expect("the sampler does not panic")
    }
}

fn run(
    threads: &[SampledThread],
    interval: Duration,
    clock: fn() -> f64,
    stop: &AtomicBool,
    method: &'static str,
) -> Samples {
    let opened: Vec<Option<os::Open>> = threads.iter().map(|thread| os::open(thread.id)).collect();
    let mut out: Vec<ThreadSamples> = threads
        .iter()
        .map(|thread| ThreadSamples {
            name: thread.name.clone(),
            samples: Vec::with_capacity(4096),
            missed: 0,
        })
        .collect();
    let mut cpu: Vec<Option<u64>> = opened
        .iter()
        .map(|open| open.as_ref().and_then(os::cpu_ns))
        .collect();
    let mut frames = [0u64; MAX_FRAMES];
    while !stop.load(Ordering::Relaxed) {
        let tick = Instant::now();
        for (index, thread) in threads.iter().enumerate() {
            let (Some(open), out) = (&opened[index], &mut out[index]) else {
                out[index].missed += 1;
                continue;
            };
            let time_ms = clock();
            let depth = os::sample(open, thread.bounds, &mut frames);
            if depth == 0 {
                out.missed += 1;
                continue;
            }
            let now = os::cpu_ns(open);
            let cpu_ns = now
                .zip(cpu[index])
                .map(|(now, then)| now.saturating_sub(then));
            cpu[index] = now;
            out.samples.push(Sample {
                time_ms,
                stack: frames[..depth].to_vec(),
                cpu_ns,
            });
        }
        if let Some(rest) = interval.checked_sub(tick.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    Samples {
        threads: out,
        interval_ms: interval.as_secs_f64() * 1000.0,
        method,
    }
}

/// Follows the frame pointer chain from `fp` within the stack `bounds`,
/// after the instruction pointer `pc`: frames written into `frames`, how
/// many. Each frame record holds the caller's frame pointer, then the
/// return address.
#[cfg(unix)]
pub(crate) fn walk_frame_pointers(
    pc: u64,
    mut fp: u64,
    (low, high): (u64, u64),
    frames: &mut [u64],
) -> usize {
    if frames.is_empty() || pc == 0 {
        return 0;
    }
    frames[0] = pc;
    let mut depth = 1;
    while depth < frames.len() {
        if fp < low || fp.saturating_add(16) > high || !fp.is_multiple_of(8) {
            break;
        }
        // SAFETY: both words lie within the stack's bounds, checked above.
        let (next, ret) = unsafe { (*(fp as *const u64), *((fp + 8) as *const u64)) };
        if ret == 0 {
            break;
        }
        frames[depth] = ret;
        depth += 1;
        // Frames go up the stack; a pointer that does not was not one.
        if next <= fp {
            break;
        }
        fp = next;
    }
    depth
}

/// `length` bytes of this program's code from `start`, for disassembly: the
/// code of a function the sampler found running, which its module maps
/// readable.
#[cfg(target_arch = "x86_64")]
pub(crate) fn code(start: u64, length: usize) -> Vec<u8> {
    if start == 0 {
        return Vec::new();
    }
    // SAFETY: `start` is a sampled function's start in a loaded module, and
    // callers stop within a few bytes past its last sampled instruction,
    // inside the module's mapped code.
    unsafe { std::slice::from_raw_parts(start as *const u8, length) }.to_vec()
}
