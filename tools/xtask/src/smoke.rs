use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

use crate::Failure;
use crate::pipeline::QUICK;
use crate::process::{Project, engine_commit, project_option, run};

pub const PLAYER: &str = "crates/engine_player";

const USAGE: &str = "usage: cargo xtask smoke [NAME...] --limit-ms MS [--project DIR] [--native-profile PROFILE] | [--project DIR] --list";

struct Options {
    project: String,
    names: Vec<String>,
    list: bool,
    profile: String,
    limit_ms: Option<String>,
}

fn options(arguments: &[String]) -> Result<Options, String> {
    let (project, rest) = project_option(arguments)?;
    let mut options = Options {
        project: project.unwrap_or_else(|| PLAYER.to_owned()),
        names: Vec::new(),
        list: false,
        profile: QUICK.to_owned(),
        limit_ms: None,
    };
    let mut given = rest.into_iter();
    while let Some(option) = given.next() {
        match option.as_str() {
            "--list" => options.list = true,
            "--native-profile" => {
                options.profile = given
                    .next()
                    .filter(|profile| !profile.starts_with("--"))
                    .ok_or(USAGE)?;
            }
            "--limit-ms" => {
                options.limit_ms = Some(
                    given
                        .next()
                        .filter(|limit| limit.parse::<u64>().is_ok_and(|limit| limit > 0))
                        .ok_or(USAGE)?,
                );
            }
            flag if flag.starts_with("--") => return Err(format!("{flag}\n{USAGE}")),
            _ => options.names.push(option),
        }
    }
    if options.list && (!options.names.is_empty() || options.limit_ms.is_some()) {
        return Err(USAGE.to_owned());
    }
    Ok(options)
}

pub fn smoke(root: &Path, arguments: &[String]) -> Result<(), Failure> {
    let options = options(arguments)?;
    let project = Project::read(root, &options.project)?;
    let program = build(root, &project, &options.profile)?;
    if options.list {
        return Ok(run(root, &program, &["--smoke", engine::smokes::LIST])?);
    }
    let names = if options.names.is_empty() {
        engine::smokes::ALL.to_owned()
    } else {
        options.names.join(",")
    };
    let limit = options.limit_ms.ok_or_else(|| USAGE.to_owned())?;
    let mut command = Command::new(&program);
    command
        .args(["--smoke", &names, "--limit-ms", &limit])
        .envs([engine_commit()])
        .current_dir(root);
    judged(&collect(command, &program)?)
}

pub fn build(root: &Path, project: &Project, profile: &str) -> Result<String, String> {
    run(
        root,
        "cargo",
        &[
            "build",
            "-p",
            &project.package,
            "--bin",
            &project.native,
            "--profile",
            profile,
            "--features",
            "dev",
            "--locked",
        ],
    )?;
    let folder = match profile {
        "dev" => "debug",
        other => other,
    };
    let program = root.join("target").join(folder).join(project.program());
    Ok(program.to_string_lossy().into_owned())
}

fn collect(mut command: Command, program: &str) -> Result<Vec<Value>, Failure> {
    let mut child = command
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|error| format!("{program}: {error}"))?;
    let mut records = Vec::new();
    if let Some(output) = child.stdout.take() {
        for line in BufReader::new(output).lines() {
            let line = line.map_err(|error| format!("{program}: {error}"))?;
            println!("{line}");
            if let Ok(record) = serde_json::from_str::<Value>(&line)
                && record["smoke"].is_string()
            {
                records.push(record);
            }
        }
    }
    let status = child
        .wait()
        .map_err(|error| format!("{program}: {error}"))?;
    match status.code() {
        Some(0 | 3) => Ok(records),
        _ => Err(json!({
            "code": "runtime_panicked",
            "path": program,
            "got": status.to_string(),
            "records": records.len(),
        })
        .to_string()
        .into()),
    }
}

fn judged(records: &[Value]) -> Result<(), Failure> {
    let failed: Vec<&str> = records
        .iter()
        .filter(|record| record["ok"] != true)
        .filter_map(|record| record["smoke"].as_str())
        .collect();
    println!(
        "{}",
        json!({
            "smokes": records.len(),
            "failed": failed,
        })
    );
    if failed.is_empty() {
        return Ok(());
    }
    Err(Failure::found(
        json!({ "code": "game_rule", "path": "$.smokes", "got": failed }).to_string(),
    ))
}
