#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod cook;
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
pub use engine_profiler as profiler;
pub use engine_profiler::{Pix, RenderDoc};
pub use job_cache::JobStore;
pub use jobs::{CacheOptions, JobOptions, JobPool};
pub use measure::{Method, Settle};
pub use offscreen::{Frames, Offscreen, PixRecord, ProfileRecord, Record, Recorded};
pub use options::{Argument, Arguments, HostOptions, PARAMETERS, Vendor, usage};
pub use window::{Measure, Window, WindowReport, run};

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

pub fn exit(outcome: engine::Result<()>) -> ExitCode {
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn host_error(what: &str, error: impl std::fmt::Display) -> engine::Error {
    engine::Error::new(
        engine::ErrorCode::RenderFailed,
        "$",
        format!("{what}: {error}"),
    )
}
