use std::{
    env,
    path::Path,
    process::{Command, ExitCode},
};

use xtask::{Failure, agent, compare, cook, dev, process, runs};

const ENGINE: &str = "xtask_engine";

fn main() -> ExitCode {
    match execute() {
        Ok(code) => code,
        Err(failure) => {
            eprintln!("xtask: {}", failure.message);
            ExitCode::from(failure.code)
        }
    }
}

fn execute() -> Result<ExitCode, Failure> {
    let mut arguments = env::args().skip(1);
    let task = arguments.next().unwrap_or_else(|| "help".to_owned());
    let options: Vec<_> = arguments.collect();
    let root = process::repository_root()?;
    if task == "agent" {
        return Ok(ExitCode::from(agent::agent(&root, &options)?));
    }
    if options.iter().any(|option| option == "--help") {
        return engine(&root, &task, &options);
    }
    match task.as_str() {
        "cook" => cook::cook(&root, &options)?,
        "dev" => dev::dev(&root, &options)?,
        "compare" => compare::compare(&root, &options)?,
        "runs" => runs::runs(&root, &options)?,
        _ => return engine(&root, &task, &options),
    }
    Ok(ExitCode::SUCCESS)
}

fn engine(root: &Path, task: &str, options: &[String]) -> Result<ExitCode, Failure> {
    let status = Command::new("cargo")
        .args(["run", "-p", ENGINE, "--", task])
        .args(options)
        .current_dir(root)
        .status()
        .map_err(|error| format!("cannot run cargo: {error}"))?;
    Ok(status
        .code()
        .and_then(|code| u8::try_from(code).ok())
        .map_or(ExitCode::FAILURE, ExitCode::from))
}
