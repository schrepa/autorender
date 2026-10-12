//! Native hosts of engine games, on DX12, Vulkan or Metal: a window the game
//! runs in as the page runs it, or an offscreen display tools draw into
//! without one, and the files games read from disk as the page fetches
//! them. Every native game takes the host's command line ([`HostOptions`]):
//! windows measured, offscreen frames, measured runs, captures, profiles,
//! hardware counters, vendor deep dives and development sessions, which
//! `engine_profiler` measures and captures for.
#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

#[cfg(feature = "dev")]
pub mod dev;
pub mod files;
mod input;
mod job_cache;
mod jobs;
mod measure;
mod offscreen;
mod options;
mod window;

use std::process::ExitCode;

pub use engine::GpuPreference;
/// Profiles and GPU captures, which the options run.
pub use engine_profiler as profiler;
pub use engine_profiler::{Pix, RenderDoc};
pub use job_cache::JobCache;
pub use jobs::{CacheOptions, JobOptions, JobPool};
pub use measure::{Method, Settle};
pub use offscreen::{Frames, Offscreen, PixRecord, ProfileRecord, Record, Recorded};
pub use options::{Argument, Arguments, HostOptions, PARAMETERS, USAGE, Vendor};
pub use window::{Measure, Window, WindowReport, run};

/// A native game's `main`: the host's options from `arguments` (`--help`
/// lists them), the game `build` makes run as they ask, `title` naming its
/// window, and its failure printed, as the process's exit code. Games with
/// options of their own parse them around the host's
/// ([`HostOptions::take`]) and run [`HostOptions::run`].
pub fn main<G: engine::Game, E: Into<engine::Error>>(
    title: &str,
    arguments: impl Iterator<Item = String>,
    build: impl FnOnce() -> Result<(engine::Engine, G), E>,
) -> ExitCode {
    let arguments: Vec<String> = arguments.collect();
    if arguments.iter().any(|argument| argument == "--help") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    match HostOptions::parse(arguments) {
        Ok(options) => exit(options.run(title, build)),
        Err(message) => {
            eprintln!(
                "{message}

{USAGE}"
            );
            ExitCode::from(2)
        }
    }
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
        let late = self.since.elapsed()
            > std::time::Duration::from_millis(engine::host::policy::PREPARE_LIMIT_MS);
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
        if (self.quiet < engine::host::policy::QUIET_FRAMES || after.loading) && !late {
            return Ok(false);
        }
        host.suspend();
        Ok(true)
    }
}
