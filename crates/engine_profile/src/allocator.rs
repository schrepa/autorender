#![expect(
    unsafe_code,
    reason = "implements the standard allocator's unsafe contract"
)]

use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Mutex, PoisonError};

pub const SAMPLE_EVERY: u32 = 1021;
pub const ALWAYS_SAMPLED_BYTES: usize = 65_536;

static SAMPLING: AtomicBool = AtomicBool::new(false);
type Sampler = fn(usize);

static SAMPLER: Mutex<Option<Sampler>> = Mutex::new(None);

thread_local! {
    static UNTIL_SAMPLE: Cell<u32> = const { Cell::new(SAMPLE_EVERY) };
    static SAMPLING_HERE: Cell<bool> = const { Cell::new(false) };
    static HERE: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
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

pub(crate) fn thread_heap() -> (u64, u64) {
    HERE.try_with(Cell::get).unwrap_or((0, 0))
}

fn allocated(bytes: usize) {
    _ = HERE.try_with(|here| {
        let (allocations, requested) = here.get();
        here.set((allocations.wrapping_add(1), requested.wrapping_add(bytes as u64)));
    });
    sampled(bytes);
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
// upholds the contract. Accounting allocates no memory and cannot panic; the
// optional allocation sampler runs after the wrapped allocator returned, never
// inside it, and a thread already sampling skips sampling, so nested
// allocations do not recurse into it.
unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller passes a layout of nonzero size.
        let pointer = unsafe { self.0.alloc(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller passes a layout of nonzero size.
        let pointer = unsafe { self.0.alloc_zeroed(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller passes a pointer this allocator returned with this
        // same layout, and the wrapped allocator returned it.
        unsafe { self.0.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller passes a pointer this allocator returned with this
        // same layout and a nonzero size that forms a valid layout with it.
        let grown = unsafe { self.0.realloc(pointer, layout, new_size) };
        if !grown.is_null() {
            allocated(new_size);
        }
        grown
    }
}
