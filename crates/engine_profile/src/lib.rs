#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

pub const MAX_EVENTS: usize = 65_536;

#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub name: &'static str,
    pub ms: f64,
    pub worst_ms: f64,
    pub runs: u64,
    pub allocations: u64,
}

#[cfg(feature = "enabled")]
mod allocator;
#[cfg(feature = "bricks")]
mod bricks;
#[cfg(feature = "enabled")]
mod collector;
#[cfg(feature = "enabled")]
mod rows;
#[cfg(feature = "enabled")]
pub use allocator::{
    ALWAYS_SAMPLED_BYTES, AllocationSnapshot, SAMPLE_EVERY, TrackingAllocator, allocation_snapshot,
    sample_allocations,
};
#[cfg(feature = "bricks")]
pub use bricks::{
    BrickHeap, BrickScope, CORE, MAX_BRICKS, allocation_cost_ms, brick_heap, current_brick, enter,
    install_clock, scope_cost_ms, take_scopes, take_spent,
};
#[cfg(feature = "enabled")]
pub use collector::*;
#[cfg(feature = "enabled")]
pub use rows::{
    RowScope, frame_mark, mark_speed, now_ms, row, take_rows, take_slow, take_speed, take_waits,
    tally, timed, waited,
};

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
    pub struct RowScope;
    impl Drop for RowScope {
        #[inline(always)]
        fn drop(&mut self) {}
    }
    #[inline(always)]
    pub fn row(_: &'static str) -> usize {
        0
    }
    #[inline(always)]
    pub fn timed(_: usize) -> RowScope {
        RowScope
    }
    #[inline(always)]
    pub fn tally(_: &'static str, _: u64) {}
    #[inline(always)]
    pub fn now_ms() -> f64 {
        0.0
    }
    #[inline(always)]
    pub fn waited(_: &str, _: f64) {}
    #[inline(always)]
    pub fn mark_speed() {}
    pub fn take_rows() -> Vec<crate::Row> {
        Vec::new()
    }
    pub fn take_waits() -> Vec<crate::Row> {
        Vec::new()
    }
    pub fn take_speed() -> Option<(f64, f64)> {
        None
    }
    #[inline(always)]
    pub fn frame_mark() -> u64 {
        0
    }
    pub fn take_slow() -> Vec<(u64, &'static str, f64)> {
        Vec::new()
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
