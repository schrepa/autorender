//! Profiles and GPU captures of native engine games, which hosts
//! (`engine_native`) run when their options ask:
//!
//! - in every build, frame-exact captures of a frame's GPU work by graphics
//!   debuggers: PIX on D3D12 ([`Pix`]) and RenderDoc ([`RenderDoc`]), and
//!   the deepest a thread's stack went ([`stack`]);
//! - with the `profiling` feature, deep profiles on any machine without
//!   privileges or setup ([`DeepCapture`]: CPU samples of the game's
//!   threads symbolicated in process, the engine's spans, every frame's GPU
//!   passes and draws, Firefox Profiler files and a summary with what the
//!   machine could add), each pass's hardware counters where drivers offer
//!   them ([`counters`]), and deep dives through the GPU vendor's tools
//!   ([`vendor`]);
//! - with the `load` feature, what other programs load the machine with
//!   while a measuring tool times a run ([`load`]).
//!
//! Every unsafe call the native tools make is here, allowed item by item.

#[cfg(feature = "profiling")]
mod capabilities;
#[cfg(all(feature = "profiling", not(target_vendor = "apple")))]
pub mod counters;
/// Hardware counters per pass count through Vulkan, which Apple's GPUs
/// draw through Metal in place of: Xcode's Metal tools count there.
#[cfg(all(feature = "profiling", target_vendor = "apple"))]
pub mod counters {
    /// Which counters to count.
    #[derive(Clone, Debug)]
    pub enum Selection {
        First,
        Named(Vec<String>),
    }

    pub fn enable(_: Selection) -> Result<(), String> {
        Err(
            "hardware counters count through Vulkan; on Apple's GPUs, Xcode's Metal \
             tools (Instruments' Metal System Trace) count"
                .into(),
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
pub use deep::{DeepCapture, ProfileFiles};
pub use pix::Pix;
pub use renderdoc::RenderDoc;
