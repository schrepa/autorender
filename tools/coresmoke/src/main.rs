mod caches;
mod content;
mod random;
mod streaming;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use serde_json::{Value, json};

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
