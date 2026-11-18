#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

#[cfg(feature = "profiling")]
mod deep;
#[cfg(feature = "profiling")]
pub mod sampler;

#[cfg(feature = "profiling")]
pub use deep::{DeepCapture, DeepFiles, DeepSizes, deep_setting, deep_sizes};
