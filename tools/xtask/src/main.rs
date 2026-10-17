//! The repository's tasks: building, serving, checking and measuring the
//! engine and its games (`cargo xtask help` lists them).

mod ambiguities;
mod authoring;
mod bench;
mod checkouts;
mod checks;
mod commands;
mod coverage;
mod development;
mod equivalence;
mod exports;
mod features;
mod gpu;
mod heavy;
mod importing;
mod iteration_build;
mod lint;
mod locality;
mod measure;
mod measure_host;
mod pipeline;
mod process;
mod profiling;
mod pure;
mod scaffold;
mod selfcheck;
mod serve;
mod shader_report;
mod smoke;
mod structure;
mod verify;

use std::{env, process::ExitCode};

/// How a command failed, and the exit code it ends with: 1 when it could
/// not do what it was asked, 3 when it did and found what it checks wrong
/// (a smoke's failed assertion), as the smokes' own exit codes say.
pub struct Failure {
    pub message: String,
    pub code: u8,
}

impl Failure {
    /// A check that ran and found what it checks wrong: exit code 3.
    pub fn found(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: 3,
        }
    }
}

impl From<String> for Failure {
    fn from(message: String) -> Self {
        Self { message, code: 1 }
    }
}

impl From<&str> for Failure {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}

fn main() -> ExitCode {
    let result = execute();
    checkouts::release();
    if let Ok(root) = process::repository_root() {
        xtask::cache::finish(&root);
    }
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            eprintln!("xtask: {}", failure.message);
            ExitCode::from(failure.code)
        }
    }
}

fn execute() -> Result<(), Failure> {
    let mut arguments = env::args().skip(1);
    let task = arguments.next().unwrap_or_else(|| "help".to_owned());
    let options: Vec<_> = arguments.collect();
    let root = process::repository_root()?;
    if task == "tooling-child" {
        let code = xtask::tooling_check::child(&options)?;
        return if code == 0 {
            Ok(())
        } else {
            Err(Failure {
                code,
                message: "tooling child failed".into(),
            })
        };
    }
    if task == "prepare-binaries" {
        return Ok(xtask::builds::prepare_binaries(&root, &options)?);
    }
    if task == "prepare-native" || task == "run-native" {
        let code = xtask::native::command(&root, &options, task == "run-native")?;
        return if code == 0 {
            Ok(())
        } else {
            Err(Failure {
                code,
                message: "native command failed".into(),
            })
        };
    }
    if task == "dev" {
        return Ok(pipeline::dev(&root, dev_port(&options)?)?);
    }
    if task == "agent" {
        return Ok(development::agent(&root, &options)?);
    }
    if task == "scene" {
        return Ok(development::scene(&options)?);
    }
    if task == "asset" {
        return Ok(importing::asset(&options)?);
    }
    if task == "profile" {
        return Ok(match options.as_slice() {
            [] => profiling::build(&root, false),
            [option] if option == "--attribution" => profiling::build(&root, true),
            _ => Err("usage: profile [--attribution]".into()),
        }?);
    }
    if task == "smoke" {
        return smoke::smoke(&root, &options);
    }
    if task == "pure" {
        return pure::pure(&root, &options);
    }
    if task == "core-smoke" {
        let mut corpus = vec!["caches".to_owned()];
        corpus.extend(options);
        return pure::pure(&root, &corpus);
    }
    if task == "bench" {
        return Ok(bench::bench(&root, &options)?);
    }
    if task == "measure" {
        return Ok(measure::measure(&root, &options)?);
    }
    if task == "equivalence" {
        return equivalence::equivalence(&root, &options);
    }
    if task == "selfcheck" {
        return selfcheck::selfcheck(&root, &options);
    }
    if task == "verify" {
        return verify::verify(&root, &options);
    }
    if task == "checks" {
        return Ok(checks::check_task(&root, &options)?);
    }
    if task == "new" {
        return Ok(scaffold::new(&root, &options)?);
    }
    if task == "gpu" && !options.is_empty() {
        return Ok(shader_report::run(&root, &options)?);
    }
    if task == "build-dev" {
        let [option, output] = options.as_slice() else {
            return Err("usage: build-dev --output web/pkg-iterations/GENERATION".into());
        };
        if option != "--output" {
            return Err("build-dev requires --output".into());
        }
        let result = iteration_build::build(&root, std::path::Path::new(output));
        println!("{result}");
        return if result["ok"] == true {
            Ok(())
        } else {
            Err("development build failed; see JSON diagnostics".into())
        };
    }
    if matches!(task.as_str(), "help" | "--help" | "-h") {
        return Ok(commands::help(&options)?);
    }
    if let Some(argument) = options.first() {
        return Err(
            format!("unexpected argument {argument:?}; run `cargo xtask help` for usage").into(),
        );
    }
    Ok(match task.as_str() {
        "build" => pipeline::build(&root),
        "lint" => lint::run_stages(&root, lint::LINT),
        "docs" => pipeline::docs(&root),
        "assets" => authoring::validate_sources(&root),
        "schema" => authoring::write_generated(&root),
        "gpu" => gpu::check(&root),
        _ => Err(format!(
            "unknown command {task:?}; run `cargo xtask help` for usage"
        )),
    }?)
}

fn dev_port(arguments: &[String]) -> Result<u16, String> {
    match arguments {
        [] => Ok(5173),
        [option, value] if option == "--port" => value
            .parse::<u16>()
            .ok()
            .filter(|port| *port != 0)
            .ok_or_else(|| "dev --port requires an integer from 1 to 65535".to_owned()),
        _ => Err("usage: cargo xtask dev [--port N]".to_owned()),
    }
}
