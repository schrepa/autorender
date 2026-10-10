//! Bounded, opt-in main-thread capture. Disabled hooks inline to no operations.
//!
//! This crate owns no browser or GPU APIs. The composition supplies a monotonic
//! millisecond clock. Profiling builds pay allocator accounting costs even when
//! not capturing; ordinary builds contain neither collector nor allocator hooks.

/// Events one capture holds at most; tooling bounds its requests by it.
pub const MAX_EVENTS: usize = 65_536;

#[cfg(feature = "enabled")]
#[allow(unsafe_code)]
mod allocator;
#[cfg(feature = "enabled")]
mod collector;
#[cfg(feature = "enabled")]
pub use allocator::{AllocationSnapshot, TrackingAllocator, allocation_snapshot};
#[cfg(feature = "enabled")]
pub use collector::*;

#[cfg(not(feature = "enabled"))]
mod disabled {
    pub struct Span;
    impl Drop for Span {
        #[inline(always)]
        fn drop(&mut self) {}
    }
    #[inline(always)]
    pub fn span(_: &'static str, _: &'static str) -> Span {
        Span
    }
    #[inline(always)]
    pub fn counter(_: &'static str, _: f64) {}
    #[inline(always)]
    pub fn active() -> bool {
        false
    }
    #[inline(always)]
    pub fn capture_id() -> Option<u64> {
        None
    }
    #[inline(always)]
    pub fn set_frame(_: u64) {}
    #[inline(always)]
    pub fn current_frame() -> u64 {
        0
    }
    /// Accumulates hot-loop timing into one result instead of one event per item.
    pub struct Aggregate;
    impl Aggregate {
        #[inline(always)]
        pub fn new(_: &'static str, _: &'static str) -> Self {
            Self
        }
        #[inline(always)]
        pub fn measure<T>(&mut self, run: impl FnOnce() -> T) -> T {
            run()
        }
    }
}
#[cfg(not(feature = "enabled"))]
pub use disabled::*;
