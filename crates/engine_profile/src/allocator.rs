#![expect(
    unsafe_code,
    reason = "implements the standard allocator's unsafe contract"
)]

use serde::Serialize;
use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, PoisonError};

pub const SAMPLE_EVERY: u32 = 1021;
pub const ALWAYS_SAMPLED_BYTES: usize = 65_536;

static AVAILABLE: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static DEALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static REALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static REQUESTED: AtomicU64 = AtomicU64::new(0);
static FREED: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static FAILURES: AtomicU64 = AtomicU64::new(0);
static SAMPLING: AtomicBool = AtomicBool::new(false);
type Sampler = fn(usize);

static SAMPLER: Mutex<Option<Sampler>> = Mutex::new(None);

thread_local! {
    static UNTIL_SAMPLE: Cell<u32> = const { Cell::new(SAMPLE_EVERY) };
    static SAMPLING_HERE: Cell<bool> = const { Cell::new(false) };
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct AllocationSnapshot {
    pub available: bool,
    pub allocations: u64,
    pub deallocations: u64,
    pub reallocations: u64,
    pub requested_bytes: u64,
    pub freed_bytes: u64,
    pub live_bytes: u64,
    pub peak_live_bytes: u64,
    pub failures: u64,
}

pub fn allocation_snapshot() -> AllocationSnapshot {
    AllocationSnapshot {
        available: AVAILABLE.load(Relaxed),
        allocations: ALLOCATIONS.load(Relaxed),
        deallocations: DEALLOCATIONS.load(Relaxed),
        reallocations: REALLOCATIONS.load(Relaxed),
        requested_bytes: REQUESTED.load(Relaxed),
        freed_bytes: FREED.load(Relaxed),
        live_bytes: LIVE.load(Relaxed),
        peak_live_bytes: PEAK.load(Relaxed),
        failures: FAILURES.load(Relaxed),
    }
}

pub fn sample_allocations(sampler: Option<fn(usize)>) {
    *SAMPLER.lock().unwrap_or_else(PoisonError::into_inner) = sampler;
    SAMPLING.store(sampler.is_some(), Relaxed);
}

pub struct TrackingAllocator<A>(A);
impl<A> TrackingAllocator<A> {
    pub const fn new(inner: A) -> Self {
        Self(inner)
    }
}

fn allocated(bytes: usize) {
    REQUESTED.fetch_add(bytes as u64, Relaxed);
    let live = LIVE
        .fetch_add(bytes as u64, Relaxed)
        .wrapping_add(bytes as u64);
    PEAK.fetch_max(live, Relaxed);
}
fn freed(bytes: usize) {
    FREED.fetch_add(bytes as u64, Relaxed);
    LIVE.fetch_sub(bytes as u64, Relaxed);
}

fn sampled(bytes: usize) {
    if !SAMPLING.load(Relaxed) {
        return;
    }
    let due = UNTIL_SAMPLE
        .try_with(|until| {
            let left = until.get().saturating_sub(1);
            until.set(if left == 0 { SAMPLE_EVERY } else { left });
            left == 0
        })
        .unwrap_or(false);
    if !(due || bytes >= ALWAYS_SAMPLED_BYTES)
        || SAMPLING_HERE
            .try_with(|here| here.replace(true))
            .unwrap_or(true)
    {
        return;
    }
    let sampler = SAMPLER.try_lock().ok().and_then(|sampler| *sampler);
    if let Some(sampler) = sampler {
        sampler(bytes);
    }
    SAMPLING_HERE.set(false);
}

// SAFETY: every call is forwarded unchanged to the wrapped allocator, which
// upholds the contract. Null failures do not affect live counts. Accounting
// allocates no memory and cannot panic; the optional allocation sampler runs
// after the wrapped allocator returned, never inside it, and a thread already
// sampling skips sampling, so nested allocations do not recurse into it.
unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        AVAILABLE.store(true, Relaxed);
        // SAFETY: the caller passes a layout of nonzero size.
        let pointer = unsafe { self.0.alloc(layout) };
        if pointer.is_null() {
            FAILURES.fetch_add(1, Relaxed);
            return pointer;
        }
        ALLOCATIONS.fetch_add(1, Relaxed);
        allocated(layout.size());
        sampled(layout.size());
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        AVAILABLE.store(true, Relaxed);
        // SAFETY: the caller passes a layout of nonzero size.
        let pointer = unsafe { self.0.alloc_zeroed(layout) };
        if pointer.is_null() {
            FAILURES.fetch_add(1, Relaxed);
            return pointer;
        }
        ALLOCATIONS.fetch_add(1, Relaxed);
        allocated(layout.size());
        sampled(layout.size());
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller passes a pointer this allocator returned with this
        // same layout, and the wrapped allocator returned it.
        unsafe { self.0.dealloc(pointer, layout) };
        DEALLOCATIONS.fetch_add(1, Relaxed);
        freed(layout.size());
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller passes a pointer this allocator returned with this
        // same layout and a nonzero size that forms a valid layout with it.
        let grown = unsafe { self.0.realloc(pointer, layout, new_size) };
        if grown.is_null() {
            FAILURES.fetch_add(1, Relaxed);
            return grown;
        }
        REALLOCATIONS.fetch_add(1, Relaxed);
        freed(layout.size());
        allocated(new_size);
        sampled(new_size);
        grown
    }
}
