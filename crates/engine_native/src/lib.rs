//! Native hosts of engine games, on DX12, Vulkan or Metal: a window the game
//! runs in as the page runs it, or an offscreen display tools draw into
//! without one, and the files games read from disk as the page fetches
//! them.

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
        Err("hardware counters count through Vulkan; on Apple's GPUs, Xcode's Metal              tools (Instruments' Metal System Trace) count"
            .into())
    }

    pub(crate) fn begin() {}

    pub(crate) fn end() -> Option<serde_json::Value> {
        None
    }
}
#[cfg(feature = "profiling")]
mod deep;
#[cfg(feature = "dev")]
pub mod dev;
pub mod files;
mod input;
mod job_cache;
mod jobs;
mod offscreen;
mod pix;
mod renderdoc;
#[cfg(feature = "profiling")]
pub mod sampler;
#[cfg(feature = "profiling")]
pub mod vendor;
mod window;

use std::process::ExitCode;

#[cfg(feature = "profiling")]
pub use deep::{DeepCapture, ProfileFiles};
pub use engine::GpuPreference;
pub use job_cache::JobCache;
pub use jobs::{CacheOptions, JobOptions, JobPool};
pub use offscreen::Offscreen;
pub use pix::Pix;
pub use renderdoc::RenderDoc;
pub use window::{Measure, Window, WindowReport, run};

/// [`run`] as a program's `main`: the game's failure printed, as the
/// process's exit code.
pub fn main<G: engine::Game, E: Into<engine::Error>>(
    window: Window,
    build: impl FnOnce() -> Result<(engine::Engine, G), E>,
) -> ExitCode {
    exit(run(window, build).map(drop))
}

/// `outcome` as a program's exit code, its failure printed.
pub fn exit(outcome: engine::Result<()>) -> ExitCode {
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

/// The failure of the platform's windowing or GPU, as `what` failed.
fn host_error(what: &str, error: impl std::fmt::Display) -> engine::Error {
    engine::Error::new(
        engine::ErrorCode::RenderFailed,
        "$",
        format!("{what}: {error}"),
    )
}

/// How a host starts showing a game, as the page does: frames draw the
/// starting state without advancing time until they stop creating
/// pipelines and terrains have loaded the view (or a minute passed), then
/// play starts. While the GPU compiles pipelines, loading goes on instead.
pub(crate) struct Preparation {
    since: std::time::Instant,
    quiet: u32,
}

// Frames without new pipelines before play starts.
const QUIET_FRAMES: u32 = 3;
const PREPARE_LIMIT: std::time::Duration = std::time::Duration::from_secs(60);

impl Preparation {
    pub(crate) fn new() -> Self {
        Self {
            since: std::time::Instant::now(),
            quiet: 0,
        }
    }

    /// One preparing frame of `host` at `timestamp` on a `size` view;
    /// whether play may start.
    pub(crate) fn frame<G: engine::Game>(
        &mut self,
        host: &mut engine::GameHost<G>,
        timestamp: f64,
        (width, height): (u32, u32),
    ) -> engine::Result<bool> {
        let late = self.since.elapsed() > PREPARE_LIMIT;
        if host.frame_stats().pipelines_compiling && !late {
            host.work(width, height);
            return Ok(false);
        }
        host.suspend();
        host.frame(
            timestamp,
            width,
            height,
            engine::InputState::default(),
            0,
            0,
            0,
        )?;
        let after = host.frame_stats();
        self.quiet = if after.render.pipelines_created == 0 {
            self.quiet + 1
        } else {
            0
        };
        if (self.quiet < QUIET_FRAMES || after.loading) && !late {
            return Ok(false);
        }
        host.suspend();
        Ok(true)
    }
}
