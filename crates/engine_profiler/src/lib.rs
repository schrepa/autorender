#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

#[cfg(feature = "profiling")]
mod capabilities;
#[cfg(all(feature = "profiling", not(target_vendor = "apple")))]
pub mod counters;
#[cfg(all(feature = "profiling", target_vendor = "apple"))]
pub mod counters {
    #[derive(Clone, Debug)]
    pub enum Selection {
        First,
        Named(Vec<String>),
    }

    pub fn enable(_: Selection) -> Result<(), engine::Error> {
        Err(
            engine::Error::new(engine::ErrorCode::ProfilingUnavailable, "$")
                .expected(engine::Expected::Supported),
        )
    }

    pub fn enabled() -> bool {
        false
    }

    pub(crate) fn begin() {}

    pub(crate) fn end() -> Option<serde_json::Value> {
        None
    }
}
#[cfg(feature = "profiling")]
mod deep;
#[cfg(feature = "load")]
pub mod load;
mod pix;
mod renderdoc;
#[cfg(feature = "profiling")]
pub mod sampler;
pub mod stack;
#[cfg(feature = "profiling")]
pub mod vendor;

#[cfg(feature = "profiling")]
pub use deep::{DeepCapture, ProfileFiles, sample_heap_from_launch};
pub use pix::Pix;
pub use renderdoc::RenderDoc;
