use std::{
    env,
    path::Path,
    process::{Command, ExitCode},
};

use xtask::{Failure, cook, iteration_build, look, pipeline, process};

const ENGINE: &str = "xtask_engine";

const BUILD_DEV: &str = "usage: build-dev --output web/pkg-iterations/GENERATION --game NAME";

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
        "build-dev" => build_dev(&root, &options)?,
        "agent" if !matches!(options.first().map(String::as_str), Some("help" | "check")) => {
            agent(&root, &options)?;
        }
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
    if arguments.first().is_some_and(|verb| verb == "iterate") {
        pipeline::install_web_dependencies(root)?;
    }
    let status = Command::new("node")
        .arg("web/scripts/agent.mjs")
        .args(arguments)
        .env(
            "ENGINE_XTASK_BIN",
            env::current_exe().map_err(|error| error.to_string())?,
        )
        .current_dir(root)
        .status()
        .map_err(|error| format!("cannot start agent bridge: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("agent command failed; see its JSON result".into())
    }
}

fn build_dev(root: &Path, options: &[String]) -> Result<(), Failure> {
    let [output_option, output, game_option, game] = options else {
        return Err(BUILD_DEV.into());
    };
    if output_option != "--output" || game_option != "--game" {
        return Err(BUILD_DEV.into());
    }
    let result = iteration_build::build(root, Path::new(output), game);
    println!("{result}");
    if result["ok"] == true {
        Ok(())
    } else {
        Err("development build failed; see JSON diagnostics".into())
    }
}
