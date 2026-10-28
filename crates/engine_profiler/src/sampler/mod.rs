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

const MAX_FRAMES: usize = 256;

pub fn method() -> Option<&'static str> {
    os::METHOD
}

#[derive(Clone)]
pub struct SampledThread {
    pub name: String,
    id: os::Id,
    bounds: (u64, u64),
}

impl SampledThread {
    pub fn current(name: impl Into<String>) -> Option<Self> {
        let (id, bounds) = os::current()?;
        Some(Self {
            name: name.into(),
            id,
            bounds,
        })
    }
}

pub struct Sample {
    pub time_ms: f64,
    pub stack: Vec<u64>,
    pub cpu_ns: Option<u64>,
}

pub struct ThreadSamples {
    pub name: String,
    pub samples: Vec<Sample>,
    pub missed: u64,
}

pub struct Samples {
    pub threads: Vec<ThreadSamples>,
    pub interval_ms: f64,
    pub method: &'static str,
}

pub struct Sampler {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<Samples>,
}

impl Sampler {
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
        if next <= fp {
            break;
        }
        fp = next;
    }
    depth
}

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
