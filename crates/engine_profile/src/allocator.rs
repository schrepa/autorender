//! The sole unsafe module: forwards each allocation operation unchanged to A.
//! Counters measure requested bytes, not allocator metadata, committed Wasm
//! pages, JS heaps or GPU memory. They include the profiler and all Rust code.

#![allow(unsafe_code)]

use serde::Serialize;
use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

static AVAILABLE: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static DEALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static REALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static REQUESTED: AtomicU64 = AtomicU64::new(0);
static FREED: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static FAILURES: AtomicU64 = AtomicU64::new(0);

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

/// Opt-in wrapper selected by the executable, never imposed by the engine.
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

// SAFETY: every call, pointer and layout is forwarded unchanged to the wrapped
// allocator. Null failures do not affect live counts. Accounting allocates no
// memory, performs no logging, and cannot re-enter the allocator or panic.
unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        AVAILABLE.store(true, Relaxed);
        // SAFETY: caller provides GlobalAlloc's required valid layout.
        let pointer = unsafe { self.0.alloc(layout) };
        if pointer.is_null() {
            FAILURES.fetch_add(1, Relaxed);
        } else {
            ALLOCATIONS.fetch_add(1, Relaxed);
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        AVAILABLE.store(true, Relaxed);
        // SAFETY: identical layout and zero-initialization contract forwarded.
        let pointer = unsafe { self.0.alloc_zeroed(layout) };
        if pointer.is_null() {
            FAILURES.fetch_add(1, Relaxed);
        } else {
            ALLOCATIONS.fetch_add(1, Relaxed);
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: pointer and original allocation layout are forwarded unchanged.
        unsafe { self.0.dealloc(pointer, layout) };
        DEALLOCATIONS.fetch_add(1, Relaxed);
        freed(layout.size());
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: original pointer/layout and required nonzero size forwarded.
        let result = unsafe { self.0.realloc(pointer, layout, new_size) };
        if result.is_null() {
            FAILURES.fetch_add(1, Relaxed);
        } else {
            REALLOCATIONS.fetch_add(1, Relaxed);
            freed(layout.size());
            allocated(new_size);
        }
        result
    }
}
