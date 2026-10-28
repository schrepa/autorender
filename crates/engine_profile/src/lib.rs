#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

pub const MAX_EVENTS: usize = 65_536;

#[cfg(feature = "enabled")]
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
