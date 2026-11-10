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
#[cfg(feature = "profiling")]
mod live;
mod offscreen;
mod options;
#[cfg(feature = "dev")]
mod pure;
#[cfg(feature = "dev")]
pub mod smoke;
mod window;

use std::process::ExitCode;

pub use engine::GpuPreference;
pub use engine_profiler as profiler;
pub use engine_profiler::{Pix, RenderDoc};
pub use job_cache::JobStore;
pub use jobs::{CacheOptions, JobOptions, JobPool, Written};
pub use offscreen::{Frames, Offscreen, PixRecord, ProfileRecord, Record, Recorded};
pub use options::{
    Argument, Arguments, HostOptions, StartScene, Vendor, open_scene, options, parameters,
};
pub use window::{Measure, Window, WindowReport, run};

pub fn main<G: engine::Game>(
    game: engine::GameEntry<G>,
    arguments: impl Iterator<Item = String>,
) -> ExitCode {
    let arguments: Vec<String> = arguments.collect();
    if let Some(help) = engine::command_line::help(&arguments, &options()) {
        println!("{help}");
        return ExitCode::SUCCESS;
    }
    match HostOptions::parse(arguments) {
        Ok(options) => {
            #[cfg(feature = "dev")]
            if let Some(names) = &options.smoke {
                return smoke::of_game(&options, names, game);
            }
            exit(options.run(game))
        }
        Err(error) => {
            let record = serde_json::json!({"event": "arguments_rejected", "error": error});
            eprintln!("{record}");
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

fn host_error(code: engine::ErrorCode, path: &str, error: impl std::fmt::Display) -> engine::Error {
    engine::Error::new(code, path).got(error.to_string())
}

pub fn file_error(path: &str, file: &std::path::Path, error: &std::io::Error) -> engine::Error {
    let code = if error.kind() == std::io::ErrorKind::NotFound {
        engine::ErrorCode::MissingFile
    } else {
        engine::ErrorCode::SourceIo
    };
    engine::Error::new(code, path).got(file.display().to_string())
}
