use std::{
    env,
    path::Path,
    process::{Command, ExitCode},
};

use xtask::{Failure, cook, dev, look, pipeline, process};

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
    if options.iter().any(|option| option == "--help") {
        return engine(&root, &task, &options);
    }
    match task.as_str() {
        "look" => look::look(&root, &options)?,
        "cook" => cook::cook(&root, &options)?,
        "dev" => dev::dev(&root, &options)?,
        "agent" => agent(&root, &options)?,
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

fn agent(root: &Path, arguments: &[String]) -> Result<(), Failure> {
    let arguments = native_program(root, arguments)?;
    let status = Command::new("node")
        .arg("web/scripts/agent.mjs")
        .args(&arguments)
        .current_dir(root)
        .status()
        .map_err(|error| format!("cannot start agent bridge: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("agent command failed; see its JSON result".into())
    }
}

fn native_program(root: &Path, arguments: &[String]) -> Result<Vec<String>, String> {
    let mut arguments = arguments.to_vec();
    let value = |name: &str| {
        arguments
            .iter()
            .position(|argument| argument == name)
            .map(|at| at + 1)
    };
    let verb = arguments.first().map(String::as_str);
    let (Some("serve"), Some(at)) = (verb, value("--native")) else {
        return Ok(arguments);
    };
    let switch = value("--profiling").and_then(|at| arguments.get(at));
    let profiling = match switch.map(String::as_str) {
        None | Some("off") => false,
        Some("on") => true,
        Some(other) => {
            return Err(serde_json::json!({
                "code": "invalid_arguments",
                "path": "--profiling",
                "got": other,
                "expected": "on|off",
            })
            .to_string());
        }
    };
    let Some(client) = arguments.get(at).and_then(|name| {
        process::clients().iter().find(|client| {
            client.native.is_some()
                && (client.package == *name || client.native.as_ref() == Some(name))
        })
    }) else {
        return Ok(arguments);
    };
    let project = process::Project::read(root, &client.folder)?;
    arguments[at] = if profiling {
        pipeline::build_profiling(root, &project)?
    } else {
        pipeline::build_native(root, &project)?
    };
    Ok(arguments)
}
