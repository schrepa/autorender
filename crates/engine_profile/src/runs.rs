use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[cfg(not(feature = "profiling"))]
const UNCOUNTED_HEAP: &str = "heap_counted_in_profiling_builds_only";

static CLOCK: OnceLock<fn() -> f64> = OnceLock::new();

#[cfg(feature = "profiling")]
pub(crate) fn now() -> Option<f64> {
    CLOCK.get().map(|clock| clock())
}

pub fn install_clock(clock: fn() -> f64) {
    CLOCK.get_or_init(|| clock);
}

#[derive(Default)]
pub struct Runs {
    runs: AtomicU64,
    nanos: AtomicU64,
    #[cfg(feature = "profiling")]
    allocations: AtomicU64,
    #[cfg(feature = "profiling")]
    bytes: AtomicU64,
}

impl Runs {
    pub fn runs(&self) -> Result<u64, &'static str> {
        Ok(self.runs.load(Relaxed))
    }

    pub fn ns(&self) -> Result<u64, &'static str> {
        if CLOCK.get().is_none() {
            return Err("clock_not_installed");
        }
        Ok(self.nanos.load(Relaxed))
    }

    #[cfg(feature = "profiling")]
    pub fn allocations(&self) -> Result<u64, &'static str> {
        Ok(self.allocations.load(Relaxed))
    }

    #[cfg(feature = "profiling")]
    pub fn allocated_bytes(&self) -> Result<u64, &'static str> {
        Ok(self.bytes.load(Relaxed))
    }

    #[cfg(not(feature = "profiling"))]
    pub fn allocations(&self) -> Result<u64, &'static str> {
        Err(UNCOUNTED_HEAP)
    }

    #[cfg(not(feature = "profiling"))]
    pub fn allocated_bytes(&self) -> Result<u64, &'static str> {
        Err(UNCOUNTED_HEAP)
    }
}

pub fn run<T>(
    runs: &Runs,
    category: &'static str,
    name: &'static str,
    work: impl FnOnce() -> T,
) -> T {
    runs.runs.fetch_add(1, Relaxed);
    let Some(&clock) = CLOCK.get() else {
        return work();
    };
    let _span = crate::span(category, name);
    #[cfg(feature = "profiling")]
    let heap = crate::allocator::thread_heap();
    let start = clock();
    let done = work();
    let end = clock();
    #[cfg(feature = "profiling")]
    {
        let (allocations, bytes) = crate::allocator::thread_heap();
        runs.allocations
            .fetch_add(allocations.wrapping_sub(heap.0), Relaxed);
        runs.bytes.fetch_add(bytes.wrapping_sub(heap.1), Relaxed);
    }
    runs.nanos
        .fetch_add(((end - start).max(0.0) * 1e6) as u64, Relaxed);
    done
}
