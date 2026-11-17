#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

#[cfg(feature = "profiling")]
mod capabilities;
#[cfg(feature = "profiling")]
mod deep;
#[cfg(feature = "profiling")]
pub mod sampler;
pub mod stack;

#[cfg(feature = "profiling")]
pub use deep::{DeepCapture, ProfileFiles};
