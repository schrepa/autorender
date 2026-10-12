//! The core smoke: the engine's incremental caches against fresh engines'
//! (hierarchy, interpolated presentation and the retained render snapshot
//! under seeded random edits), terrain streaming without a clock or
//! workers as a deterministic trace, and the Stress Lab's content as
//! digests. GPU-free, and built in the dev profile so the engine's debug
//! assertions run. `cargo xtask core-smoke` (the caches corpus of
//! `cargo xtask pure`) runs it and holds what it observes to goldens.
//!
//! ```text
//! cargo run -p coresmoke -- [--seed N] [--assets FOLDER] [--trace FILE]
//! ```
//!
//! `--assets` names the folder of the generated files the dense and
//! highlands scenes read (`cargo xtask bench assets` prints it); without
//! it they are left out. Prints one JSON object:
//! `{"observations": {...}, "broken": [...]}`. Exits 3 when a cache
//! disagreed with a fresh engine's (its entry names the seed and the
//! operations since the caches last agreed), 1 when it could not run.

mod caches;
mod content;
mod random;
mod streaming;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use serde_json::{Value, json};

/// What the smoke observed, by name, and the promises it found broken.
#[derive(Default)]
pub struct Report {
    observations: BTreeMap<String, Value>,
    broken: Vec<String>,
}

impl Report {
    pub fn record(&mut self, name: impl Into<String>, value: impl Into<Value>) {
        self.observations.insert(name.into(), value.into());
    }

    pub fn broke(&mut self, what: String) {
        self.broken.push(what);
    }
}

/// The first 16 hexadecimal digits of `bytes`' content hash.
pub fn hash(bytes: &[u8]) -> String {
    format!("{:016x}", engine_core::content_hash(bytes) as u64)
}

struct Options {
    seed: u64,
    assets: Option<PathBuf>,
    trace: Option<PathBuf>,
}

fn options() -> Result<Options, String> {
    let usage = "usage: coresmoke [--seed N] [--assets FOLDER] [--trace FILE]";
    let mut options = Options {
        seed: caches::SEED,
        assets: None,
        trace: None,
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = || arguments.next().ok_or(usage);
        match argument.as_str() {
            "--seed" => {
                options.seed = value()?
                    .parse()
                    .map_err(|error| format!("--seed: {error}"))?;
            }
            "--assets" => options.assets = Some(value()?.into()),
            "--trace" => options.trace = Some(value()?.into()),
            _ => return Err(usage.into()),
        }
    }
    Ok(options)
}

fn main() -> ExitCode {
    let options = match options() {
        Ok(options) => options,
        Err(error) => {
            eprintln!("coresmoke: {error}");
            return ExitCode::FAILURE;
        }
    };
    let mut report = Report::default();
    let run = caches::run(options.seed, &mut report)
        .and_then(|()| streaming::run(options.trace.as_deref(), &mut report))
        .and_then(|()| content::run(options.assets.as_deref(), &mut report));
    if let Err(error) = run {
        eprintln!("coresmoke: {error}");
        return ExitCode::FAILURE;
    }
    println!(
        "{}",
        json!({"observations": report.observations, "broken": report.broken})
    );
    if report.broken.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(3)
    }
}
