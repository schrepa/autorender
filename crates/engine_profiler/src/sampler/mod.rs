use std::cell::RefCell;
use std::fmt::Arguments;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[cfg(windows)]
#[path = "windows.rs"]
mod os;

#[cfg(not(windows))]
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
    alive: Option<Arc<Mutex<bool>>>,
}

impl SampledThread {
    pub fn current(name: impl Into<String>) -> Option<Self> {
        let (id, bounds) = os::current()?;
        Some(Self {
            name: name.into(),
            id,
            bounds,
            alive: None,
        })
    }

    fn living(&self) -> Option<Option<MutexGuard<'_, bool>>> {
        let Some(alive) = &self.alive else {
            return Some(None);
        };
        let alive = alive.lock().unwrap_or_else(PoisonError::into_inner);
        if *alive { Some(Some(alive)) } else { None }
    }
}

static ANNOUNCED: Mutex<Vec<SampledThread>> = Mutex::new(Vec::new());

thread_local! {
    static LEAVING: RefCell<Option<Leaving>> = const { RefCell::new(None) };
}

struct Leaving(Arc<Mutex<bool>>);

impl Drop for Leaving {
    fn drop(&mut self) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = false;
        ANNOUNCED
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|thread| {
                !thread
                    .alive
                    .as_ref()
                    .is_some_and(|alive| Arc::ptr_eq(alive, &self.0))
            });
    }
}

pub fn listen_for_threads() {
    engine_profile::listen_to_threads(Some(announced));
}

fn announced(name: Arguments<'_>) {
    let Some(mut thread) = SampledThread::current(name.to_string()) else {
        return;
    };
    let alive = Arc::new(Mutex::new(true));
    thread.alive = Some(Arc::clone(&alive));
    if LEAVING
        .try_with(|leaving| leaving.replace(Some(Leaving(alive))))
        .is_ok()
    {
        ANNOUNCED
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(thread);
    }
}

pub fn announced_threads() -> Vec<SampledThread> {
    ANNOUNCED
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
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
        limit: usize,
    ) -> Option<Self> {
        let method = os::METHOD?;
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("engine-sampler".into())
            .spawn(move || run(&threads, (interval, limit), clock, &stopping, method))
            .ok()?;
        Some(Self { stop, thread })
    }

    pub fn finish(self) -> Samples {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.join().expect("sampler_unpanicked")
    }
}

fn run(
    threads: &[SampledThread],
    (interval, limit): (Duration, usize),
    clock: fn() -> f64,
    stop: &AtomicBool,
    method: &'static str,
) -> Samples {
    let (opened, mut cpu): (Vec<Option<os::Open>>, Vec<Option<u64>>) = threads
        .iter()
        .map(|thread| {
            let Some(_alive) = thread.living() else {
                return (None, None);
            };
            let open = os::open(thread.id);
            let cpu = open.as_ref().and_then(os::cpu_ns);
            (open, cpu)
        })
        .unzip();
    let mut out: Vec<ThreadSamples> = threads
        .iter()
        .map(|thread| ThreadSamples {
            name: thread.name.clone(),
            samples: Vec::with_capacity(limit),
            missed: 0,
        })
        .collect();
    let mut frames = [0u64; MAX_FRAMES];
    while !stop.load(Ordering::Relaxed) {
        let tick = Instant::now();
        for (index, thread) in threads.iter().enumerate() {
            let (Some(open), out) = (&opened[index], &mut out[index]) else {
                out[index].missed += 1;
                continue;
            };
            let Some(_alive) = thread.living() else {
                out.missed += 1;
                continue;
            };
            let time_ms = clock();
            let depth = os::sample(open, thread.bounds, &mut frames);
            if depth == 0 || out.samples.len() >= limit {
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
