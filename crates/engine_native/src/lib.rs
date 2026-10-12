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
pub use job_cache::JobStore;
pub use jobs::{CacheOptions, JobOptions, JobPool};
pub use measure::{Method, Settle};
pub use offscreen::{Frames, Offscreen, PixRecord, ProfileRecord, Record, Recorded};
pub use options::{Argument, Arguments, HostOptions, PARAMETERS, Vendor, usage};
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
        println!("{}", usage());
        return ExitCode::SUCCESS;
    }
    match HostOptions::parse(arguments) {
        Ok(options) => exit(options.run(title, build)),
        Err(message) => {
            eprintln!("{message}\n\n{}", usage());
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
