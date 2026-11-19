#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

#[cfg(feature = "dev")]
mod dev;
mod input;
mod io;
mod mixer;
mod offscreen;
mod options;
mod output;
mod sound;
mod window;

use std::process::ExitCode;

use offscreen::Offscreen;
use options::{HostOptions, open_scene, options};
use window::Window;

const PIPELINE_CACHE_MIB: u64 = 256;

pub fn main<G: engine::Game>(
    game: engine::GameEntry,
    arguments: impl Iterator<Item = String>,
) -> ExitCode {
    let arguments: Vec<String> = arguments.collect();
    if let Some(help) = engine::command_line::help(&arguments, &options()) {
        println!("{help}");
        return ExitCode::SUCCESS;
    }
    match HostOptions::parse(arguments) {
        Ok(options) => exit(options.run::<G>(game)),
        Err(error) => {
            let record = serde_json::json!({"event": "arguments_rejected", "error": error});
            eprintln!("{record}");
            ExitCode::from(2)
        }
    }
}

fn exit(outcome: engine::Result<()>) -> ExitCode {
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn keep_pipeline_cache() {
    let folder = std::env::current_exe()
        .ok()
        .and_then(|program| Some(program.parent()?.join("engine-pipeline-cache")));
    if let Some(folder) = folder {
        engine::keep_pipeline_cache_in(folder, PIPELINE_CACHE_MIB * 1024 * 1024);
    }
}

fn host_error(code: engine::ErrorCode, path: &str, error: impl std::fmt::Display) -> engine::Error {
    engine::Error::new(code, path).got(error.to_string())
}

fn file_error(path: &str, file: &std::path::Path, error: &std::io::Error) -> engine::Error {
    let code = if error.kind() == std::io::ErrorKind::NotFound {
        engine::ErrorCode::MissingFile
    } else {
        engine::ErrorCode::SourceIo
    };
    engine::Error::new(code, path).got(file.display().to_string())
}
