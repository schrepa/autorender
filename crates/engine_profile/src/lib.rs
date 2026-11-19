#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

#[cfg(feature = "profiling")]
mod allocator;
#[cfg(feature = "profiling")]
mod collector;
#[cfg(target_arch = "wasm32")]
mod heap;
#[cfg(feature = "dev")]
mod runs;
#[cfg(feature = "profiling")]
mod threads;
#[cfg(feature = "profiling")]
pub use allocator::{ALWAYS_SAMPLED_BYTES, SAMPLE_EVERY, TrackingAllocator, sample_allocations};
#[cfg(feature = "profiling")]
pub use collector::*;
#[cfg(target_arch = "wasm32")]
pub use heap::GrowingHeap;
#[cfg(feature = "dev")]
pub use runs::*;
#[cfg(feature = "profiling")]
pub use threads::{announce_thread, listen_to_threads};

#[cfg(not(feature = "profiling"))]
#[inline(always)]
pub fn announce_thread(_: std::fmt::Arguments<'_>) {}

#[cfg(not(feature = "profiling"))]
mod unkept {
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
    pub fn span_of(_: &'static str, _: impl std::fmt::Display) -> Span {
        Span
    }
    #[inline(always)]
    pub fn counter(_: &'static str, _: f64) {}
}
#[cfg(not(feature = "profiling"))]
pub use unkept::*;

#[cfg(not(feature = "dev"))]
mod unmeasured {
    const UNMEASURED: &str = "measured_in_dev_builds_only";

    #[inline(always)]
    pub fn install_clock(_: fn() -> f64) {}

    #[derive(Default)]
    pub struct Runs;
    impl Runs {
        pub fn runs(&self) -> Result<u64, &'static str> {
            Err(UNMEASURED)
        }
        pub fn ns(&self) -> Result<u64, &'static str> {
            Err(UNMEASURED)
        }
        pub fn allocations(&self) -> Result<u64, &'static str> {
            Err(UNMEASURED)
        }
        pub fn allocated_bytes(&self) -> Result<u64, &'static str> {
            Err(UNMEASURED)
        }
    }
    #[inline(always)]
    pub fn run<T>(_: &Runs, _: &'static str, _: &'static str, work: impl FnOnce() -> T) -> T {
        work()
    }
}
#[cfg(not(feature = "dev"))]
pub use unmeasured::*;
