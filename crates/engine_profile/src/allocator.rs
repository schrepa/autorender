//! The sole unsafe module: forwards each allocation operation unchanged to A.
//! Counters measure requested bytes, not allocator metadata, committed Wasm
//! pages, JS heaps or GPU memory. They include the profiler and all Rust code.

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::System;

    // Delegates normal operations, but deterministically rejects a realloc
    // request to verify that failure preserves the original live allocation.
    struct RejectLargeRealloc;
    // SAFETY: forwards all valid allocation contracts to System, except allowed
    // null failure without modifying the original pointer for size 128.
    unsafe impl GlobalAlloc for RejectLargeRealloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: caller's valid layout is unchanged.
            unsafe { System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            // SAFETY: caller's valid layout and zeroing contract are unchanged.
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            // SAFETY: original allocation pointer/layout are unchanged.
            unsafe { System.dealloc(pointer, layout) }
        }
        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            if size == 128 {
                return std::ptr::null_mut();
            }
            // SAFETY: original pointer/layout and requested size are unchanged.
            unsafe { System.realloc(pointer, layout, size) }
        }
    }

    #[test]
    fn allocator_forwards_zeroing_growth_shrink_and_preserves_failed_realloc() {
        // This test calls a local wrapper. The test harness/global allocator is
        // not wrapped, so concurrent collector tests cannot change counters.
        let allocator = TrackingAllocator::new(RejectLargeRealloc);
        let before = allocation_snapshot();
        let small = Layout::from_size_align(16, 8).unwrap();
        let large = Layout::from_size_align(64, 8).unwrap();
        let shrunk = Layout::from_size_align(8, 8).unwrap();
        // SAFETY: each non-null pointer is used only within its current layout;
        // successful reallocations replace the pointer, failure preserves it.
        unsafe {
            let pointer = allocator.alloc_zeroed(small);
            assert!(!pointer.is_null());
            assert!(
                std::slice::from_raw_parts(pointer, 16)
                    .iter()
                    .all(|byte| *byte == 0)
            );
            pointer.write(77);
            let pointer = allocator.realloc(pointer, small, 64);
            assert!(!pointer.is_null());
            assert_eq!(pointer.read(), 77);
            let failure_before = allocation_snapshot();
            assert!(allocator.realloc(pointer, large, 128).is_null());
            let failure_after = allocation_snapshot();
            assert_eq!(failure_after.live_bytes, failure_before.live_bytes);
            assert_eq!(
                failure_after.requested_bytes,
                failure_before.requested_bytes
            );
            assert_eq!(failure_after.freed_bytes, failure_before.freed_bytes);
            assert_eq!(pointer.read(), 77);
            let pointer = allocator.realloc(pointer, large, 8);
            assert!(!pointer.is_null());
            assert_eq!(pointer.read(), 77);
            allocator.dealloc(pointer, shrunk);
            let pointer = allocator.alloc(small);
            assert!(!pointer.is_null());
            allocator.dealloc(pointer, small);
        }
        let after = allocation_snapshot();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.allocations - before.allocations, 2);
        assert_eq!(after.deallocations - before.deallocations, 2);
        assert_eq!(after.reallocations - before.reallocations, 2);
        assert_eq!(after.requested_bytes - before.requested_bytes, 104);
        assert_eq!(after.freed_bytes - before.freed_bytes, 104);
        assert_eq!(after.failures - before.failures, 1);
        assert!(after.peak_live_bytes >= before.live_bytes + 64);
    }
}
