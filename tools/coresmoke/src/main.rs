mod caches;
mod random;
mod streaming;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use engine_core::command_line::{self, CommandOption};
use serde_json::{Value, json};

#[derive(Default)]
pub struct Report {
    observations: BTreeMap<String, Value>,
    broken: Vec<Value>,
}

impl Report {
    pub fn record(&mut self, name: impl Into<String>, value: impl Into<Value>) {
        self.observations.insert(name.into(), value.into());
    }

    pub fn broke(&mut self, broken: Value) {
        self.broken.push(broken);
    }
}

pub fn hash(bytes: &[u8]) -> String {
    format!("{:016x}", engine_core::content_hash(bytes) as u64)
}

struct Options {
    seed: u64,
    trace: Option<PathBuf>,
}

fn command_options() -> Vec<CommandOption> {
    vec![
        CommandOption::taking("--seed", "N").defaults_to(caches::SEED),
        CommandOption::taking("--trace", "FILE"),
        CommandOption::flag("--help"),
        CommandOption::flag("--json"),
    ]
}

fn options(arguments: Vec<String>) -> Result<Options, String> {
    let mut options = Options {
        seed: caches::SEED,
        trace: None,
    };
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        let mut value = || arguments.next().ok_or("--seed and --trace take a value");
        match argument.as_str() {
            "--seed" => {
                options.seed = value()?
                    .parse()
                    .map_err(|error| format!("--seed: {error}"))?;
            }
            "--trace" => options.trace = Some(value()?.into()),
            _ => return Err(format!("unknown option {argument}")),
        }
    }
    Ok(options)
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if let Some(help) = command_line::help(&arguments, &command_options()) {
        println!("{help}");
        return ExitCode::SUCCESS;
    }
    let options = match options(arguments) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{}", json!({"event": "arguments_rejected", "error": error}));
            return ExitCode::FAILURE;
        }
    };
    let mut report = Report::default();
    let run = caches::run(options.seed, &mut report)
        .and_then(|()| streaming::run(options.trace.as_deref(), &mut report));
    if let Err(error) = run {
        eprintln!("{}", json!({"event": "smoke_failed", "error": error}));
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
