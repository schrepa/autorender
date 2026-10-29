#![expect(
    unsafe_code,
    reason = "implements the standard allocator's unsafe contract"
)]

use serde::Serialize;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, PoisonError};

use crate::bricks::{
    CALIBRATION_PAIRS, HEAP_ALLOCATIONS, HEAP_LIVE, HEAP_PEAK, current_brick, slot,
};

pub const SAMPLE_EVERY: u32 = 1021;
const CALIBRATION_BLOCK: Layout = Layout::new::<[u64; 8]>();

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
type Sampler = fn(u16, usize);

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

pub fn sample_allocations(sampler: Option<fn(u16, usize)>) {
    *SAMPLER.lock().unwrap_or_else(PoisonError::into_inner) = sampler;
    SAMPLING.store(sampler.is_some(), Relaxed);
}

pub struct TrackingAllocator<A>(A);
impl<A> TrackingAllocator<A> {
    pub const fn new(inner: A) -> Self {
        Self(inner)
    }
}

fn padded(layout: Layout) -> Option<(Layout, usize)> {
    let align = layout.align().max(align_of::<u64>());
    let size = layout.size().checked_add(align)?;
    Some((Layout::from_size_align(size, align).ok()?, align))
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

fn charged(owner: u16, bytes: usize) {
    let slot = slot(owner);
    let live = HEAP_LIVE[slot]
        .fetch_add(bytes as u64, Relaxed)
        .wrapping_add(bytes as u64);
    HEAP_PEAK[slot].fetch_max(live, Relaxed);
}

fn discharged(owner: u16, bytes: usize) {
    HEAP_LIVE[slot(owner)].fetch_sub(bytes as u64, Relaxed);
}

fn counted(brick: u16, bytes: usize) {
    HEAP_ALLOCATIONS[slot(brick)].fetch_add(1, Relaxed);
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
    if !due
        || SAMPLING_HERE
            .try_with(|here| here.replace(true))
            .unwrap_or(true)
    {
        return;
    }
    let sampler = SAMPLER.try_lock().ok().and_then(|sampler| *sampler);
    if let Some(sampler) = sampler {
        sampler(brick, bytes);
    }
    SAMPLING_HERE.set(false);
}

unsafe fn owner_of(pointer: *mut u8) -> u16 {
    // SAFETY: callers pass a pointer this allocator returned: it follows a
    // header of at least eight bytes whose last four hold the owner, written
    // at allocation, and its alignment keeps that word four-aligned.
    let owner = unsafe { pointer.sub(4).cast::<u32>().read() };
    owner as u16
}

unsafe fn mark(pointer: *mut u8, owner: u16) {
    // SAFETY: callers pass a pointer at least eight bytes past the start of
    // its block, so the four bytes before it are the block's header.
    unsafe { pointer.sub(4).cast::<u32>().write(u32::from(owner)) };
}

// SAFETY: every call is forwarded to the wrapped allocator with the caller's
// layout padded by a header in front of the returned pointer: the header is the
// layout's alignment, at least eight bytes, so the pointer keeps the caller's
// alignment and the header keeps the owning brick. Deallocation and
// reallocation recompute the same padding from the caller's layout and hand the
// wrapped allocator the block it returned. Null failures do not affect live
// counts. Accounting allocates no memory and cannot panic; the optional
// allocation sampler runs after the wrapped allocator returned, never inside
// it, and a thread already sampling skips sampling, so nested allocations do
// not recurse into it.
unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        AVAILABLE.store(true, Relaxed);
        let Some((whole, header)) = padded(layout) else {
            FAILURES.fetch_add(1, Relaxed);
            return std::ptr::null_mut();
        };
        // SAFETY: `whole` is a valid nonzero layout built from the caller's.
        let block = unsafe { self.0.alloc(whole) };
        if block.is_null() {
            FAILURES.fetch_add(1, Relaxed);
            return block;
        }
        // SAFETY: the block holds `header` bytes before the caller's bytes.
        let pointer = unsafe { block.add(header) };
        let owner = current_brick();
        // SAFETY: the pointer lies `header` bytes, at least eight, into its block.
        unsafe { mark(pointer, owner) };
        ALLOCATIONS.fetch_add(1, Relaxed);
        allocated(layout.size());
        charged(owner, layout.size());
        counted(owner, layout.size());
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        AVAILABLE.store(true, Relaxed);
        let Some((whole, header)) = padded(layout) else {
            FAILURES.fetch_add(1, Relaxed);
            return std::ptr::null_mut();
        };
        // SAFETY: `whole` is a valid nonzero layout built from the caller's.
        let block = unsafe { self.0.alloc_zeroed(whole) };
        if block.is_null() {
            FAILURES.fetch_add(1, Relaxed);
            return block;
        }
        // SAFETY: the block holds `header` bytes before the caller's bytes.
        let pointer = unsafe { block.add(header) };
        let owner = current_brick();
        // SAFETY: the pointer lies `header` bytes, at least eight, into its block.
        unsafe { mark(pointer, owner) };
        ALLOCATIONS.fetch_add(1, Relaxed);
        allocated(layout.size());
        charged(owner, layout.size());
        counted(owner, layout.size());
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        let Some((whole, header)) = padded(layout) else {
            return;
        };
        // SAFETY: the caller passes a pointer this allocator returned.
        let owner = unsafe { owner_of(pointer) };
        // SAFETY: the block started `header` bytes before the pointer and was
        // allocated with `whole`, recomputed from the caller's same layout.
        unsafe { self.0.dealloc(pointer.sub(header), whole) };
        DEALLOCATIONS.fetch_add(1, Relaxed);
        freed(layout.size());
        discharged(owner, layout.size());
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let align = layout.align().max(align_of::<u64>());
        let grown = new_size
            .checked_add(align)
            .filter(|&size| Layout::from_size_align(size, align).is_ok());
        let (Some((whole, header)), Some(grown)) = (padded(layout), grown) else {
            FAILURES.fetch_add(1, Relaxed);
            return std::ptr::null_mut();
        };
        // SAFETY: the caller passes a pointer this allocator returned.
        let owner = unsafe { owner_of(pointer) };
        // SAFETY: the block started `header` bytes before the pointer and was
        // allocated with `whole`; `grown` is nonzero and forms a valid layout
        // with the block's alignment.
        let block = unsafe { self.0.realloc(pointer.sub(header), whole, grown) };
        if block.is_null() {
            FAILURES.fetch_add(1, Relaxed);
            return block;
        }
        REALLOCATIONS.fetch_add(1, Relaxed);
        freed(layout.size());
        allocated(new_size);
        discharged(owner, layout.size());
        charged(owner, new_size);
        counted(current_brick(), new_size);
        // SAFETY: the reallocated block kept its header, copied with the bytes.
        unsafe { block.add(header) }
    }
}

pub(crate) fn tracking_calibration(clock: fn() -> f64) -> f64 {
    if !AVAILABLE.load(Relaxed) {
        return 0.0;
    }
    let owner = slot(current_brick());
    let (tracked, counted) = paired(&TrackingAllocator::new(System), clock);
    let (plain, _) = paired(&System, clock);
    let bytes = counted * CALIBRATION_BLOCK.size() as u64;
    ALLOCATIONS.fetch_sub(counted, Relaxed);
    DEALLOCATIONS.fetch_sub(counted, Relaxed);
    REQUESTED.fetch_sub(bytes, Relaxed);
    FREED.fetch_sub(bytes, Relaxed);
    HEAP_ALLOCATIONS[owner].fetch_sub(counted, Relaxed);
    ((tracked - plain) / f64::from(CALIBRATION_PAIRS)).max(0.0)
}

fn paired<A: GlobalAlloc>(allocator: &A, clock: fn() -> f64) -> (f64, u64) {
    let mut done = 0;
    let start = clock();
    for _ in 0..CALIBRATION_PAIRS {
        // SAFETY: the calibration block's layout has a nonzero size.
        let pointer = std::hint::black_box(unsafe { allocator.alloc(CALIBRATION_BLOCK) });
        if !pointer.is_null() {
            // SAFETY: the pointer came from this allocator with this same layout.
            unsafe { allocator.dealloc(pointer, CALIBRATION_BLOCK) };
            done += 1;
        }
    }
    (clock() - start, done)
}
