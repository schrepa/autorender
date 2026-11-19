use std::{
    env,
    path::Path,
    process::{Command, ExitCode},
};

use xtask::{agent, dev, process};

const ENGINE: &str = "xtask_engine";

fn main() -> ExitCode {
    match execute() {
        Ok(code) => code,
        Err(failure) => {
            eprintln!("xtask: {failure}");
            ExitCode::FAILURE
        }
    }
}

fn execute() -> Result<ExitCode, String> {
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
    if task == "dev" {
        dev::dev(&root, &options)?;
        return Ok(ExitCode::SUCCESS);
    }
    engine(&root, &task, &options)
}

fn engine(root: &Path, task: &str, options: &[String]) -> Result<ExitCode, String> {
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
