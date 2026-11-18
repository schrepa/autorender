#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

#[derive(Clone, Copy, Debug)]
pub enum Fact<'a> {
    Count(u64),
    Number(f64),
    Text(&'a str),
    Unavailable(&'static str),
}

#[cfg(feature = "profiling")]
mod allocator;
#[cfg(feature = "dev")]
mod collector;
#[cfg(target_arch = "wasm32")]
mod heap;
#[cfg(feature = "profiling")]
mod threads;
#[cfg(feature = "profiling")]
pub use allocator::{ALWAYS_SAMPLED_BYTES, SAMPLE_EVERY, TrackingAllocator, sample_allocations};
#[cfg(feature = "dev")]
pub use collector::*;
#[cfg(target_arch = "wasm32")]
pub use heap::GrowingHeap;
#[cfg(feature = "profiling")]
pub use threads::{announce_thread, listen_to_threads};

#[cfg(not(feature = "profiling"))]
#[inline(always)]
pub fn announce_thread(_: std::fmt::Arguments<'_>) {}

#[cfg(not(feature = "dev"))]
mod disabled {
    const UNMEASURED: &str = "measured_in_dev_builds_only";

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
    pub fn event(_: &'static str, _: &str, _: &[(&'static str, crate::Fact<'_>)]) {}
    #[inline(always)]
    pub fn recording() -> bool {
        false
    }
    #[inline(always)]
    pub fn set_frame(_: u64) {}
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
pub use disabled::*;
