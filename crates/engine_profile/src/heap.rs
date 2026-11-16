#![expect(
    unsafe_code,
    reason = "implements the standard allocator's unsafe contract"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::arch::wasm32::memory_size;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

const PAGE_BYTES: usize = 65_536;
const HEADROOM_SHARE: usize = 2;
const BLOCK_ALIGN: usize = 16;

pub struct GrowingHeap {
    seen: AtomicUsize,
}

impl GrowingHeap {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            seen: AtomicUsize::new(0),
        }
    }

    fn reserve(&self) {
        let pages = memory_size(0);
        if pages == self.seen.load(Relaxed) {
            return;
        }
        let room = pages.saturating_mul(PAGE_BYTES) / HEADROOM_SHARE;
        if let Ok(layout) = Layout::from_size_align(room, BLOCK_ALIGN)
            && layout.size() > 0
        {
            // SAFETY: the layout has a non-zero size, as `System::alloc` requires.
            let block = unsafe { System.alloc(layout) };
            if !block.is_null() {
                // SAFETY: `block` was just returned by `System::alloc` with this same layout.
                unsafe { System.dealloc(block, layout) };
            }
        }
        self.seen.store(memory_size(0), Relaxed);
    }
}

impl Default for GrowingHeap {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: every call forwards to `System` with the caller's own pointer and layout; the
// headroom block in `reserve` is allocated and freed with one layout and never handed out.
unsafe impl GlobalAlloc for GrowingHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract, which `System` shares.
        let made = unsafe { System.alloc(layout) };
        self.reserve();
        made
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::alloc_zeroed`'s contract, which `System` shares.
        let made = unsafe { System.alloc_zeroed(layout) };
        self.reserve();
        made
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator, which takes every block from `System`.
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr` came from this allocator, which takes every block from `System`.
        let made = unsafe { System.realloc(ptr, layout, new_size) };
        self.reserve();
        made
    }
}
