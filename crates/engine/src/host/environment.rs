//! The host a report was measured on, as every host's reports name it.

use super::AdapterStatus;

/// What may differ between two hosts' reports of the same build, workload
/// and settings, everything else being the same on every host: how each
/// compiles shaders, paces and presents frames, reads back from the GPU
/// and reads ranges of files, the code each compiles to, its timers'
/// resolution, its device's limits, and Basis textures, which only pages
/// transcode.
pub const KNOWN_DIFFERENCES: &[&str] = &[
    "shader_toolchain",
    "presentation_pacing",
    "readback_latency",
    "range_read_io",
    "codegen",
    "timer_resolution",
    "device_limits",
    "basis_transcoding",
];

/// The host a report was measured on: its view, the browser or native host
/// and its GPU, with what may differ between hosts' reports
/// ([`KNOWN_DIFFERENCES`]), so a comparison across hosts holds only where
/// everything else matches.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct HostEnvironment {
    /// The view's width, in physical pixels.
    pub canvas_width: u32,
    /// The view's height, in physical pixels.
    pub canvas_height: u32,
    /// Device pixels per CSS pixel; 1 natively.
    pub device_pixel_ratio: f64,
    /// The host: the browser's user agent, or `engine_native`.
    pub user_agent: String,
    /// The GPU it drew on.
    pub adapter: AdapterStatus,
    /// What may differ from another host's report: [`KNOWN_DIFFERENCES`].
    pub known_differences: &'static [&'static str],
}
