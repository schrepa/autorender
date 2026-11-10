use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

use crate::Failure;
use crate::pipeline::build_native;
use crate::process::Project;

const USAGE: &str =
    "usage: cargo xtask pairs --limit-ms MS [--at-once N] [--project DIR]... [--scene PATH]";

pub const PROJECTS: [&str; 6] = [
    "crates/engine/examples/bare_game",
    "crates/engine/examples/walker_game",
    "crates/engine/examples/flat_game",
    "crates/engine/examples/app_template",
    "crates/engine_player",
    "examples/highlands",
];

struct Options {
    projects: Vec<String>,
    limit_ms: String,
    at_once: String,
    scene: Option<String>,
}

fn positive(value: Option<&String>) -> Result<String, String> {
    value
        .filter(|given| given.parse::<u64>().is_ok_and(|given| given > 0))
        .cloned()
        .ok_or_else(|| USAGE.to_owned())
}

fn options(arguments: &[String]) -> Result<Options, String> {
    let mut projects = Vec::new();
    let mut limit_ms = None;
    let mut at_once = None;
    let mut scene = None;
    let mut given = arguments.iter();
    while let Some(option) = given.next() {
        match option.as_str() {
            "--project" => projects.push(given.next().ok_or(USAGE)?.replace('\\', "/")),
            "--limit-ms" => limit_ms = Some(positive(given.next())?),
            "--at-once" => at_once = Some(positive(given.next())?),
            "--scene" => scene = Some(given.next().ok_or(USAGE)?.clone()),
            _ => return Err(format!("{option}\n{USAGE}")),
        }
    }
    if projects.is_empty() {
        projects = PROJECTS.map(str::to_owned).to_vec();
    }
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    Ok(Options {
        projects,
        limit_ms: limit_ms.ok_or(USAGE)?,
        at_once: at_once.unwrap_or_else(|| cores.div_ceil(2).to_string()),
        scene,
    })
}

fn started(root: &Path, program: &str, options: &Options) -> Result<Vec<Value>, Failure> {
    let mut command = Command::new(program);
    command
        .args(["--pairs", &options.at_once, "--limit-ms", &options.limit_ms])
        .current_dir(root);
    if let Some(scene) = &options.scene {
        command.args(["--scene", scene]);
    }
    let mut child = command
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|error| format!("{program}: {error}"))?;
    let mut lines = Vec::new();
    if let Some(output) = child.stdout.take() {
        for line in BufReader::new(output).lines() {
            let line = line.map_err(|error| format!("{program}: {error}"))?;
            println!("{line}");
            if let Ok(record) = serde_json::from_str::<Value>(&line)
                && record["without"].is_string()
            {
                lines.push(record);
            }
        }
    }
    let status = child
        .wait()
        .map_err(|error| format!("{program}: {error}"))?;
    match status.code() {
        Some(0 | 3) => Ok(lines),
        _ => Err(json!({
            "code": "runtime_panicked",
            "path": program,
            "got": status.to_string(),
            "starts": lines.len(),
        })
        .to_string()
        .into()),
    }
}

pub fn pairs(root: &Path, arguments: &[String]) -> Result<(), Failure> {
    let options = options(arguments)?;
    let mut failed: Vec<Value> = Vec::new();
    let mut starts = 0;
    for folder in &options.projects {
        let project = Project::read(root, folder)?;
        let program = build_native(root, &project)?;
        for line in started(root, &program, &options)? {
            starts += 1;
            if line["ok"] != true {
                failed.push(json!({ "project": folder, "without": line["without"] }));
            }
        }
    }
    println!(
        "{}",
        json!({ "pairs": options.projects, "starts": starts, "failed": failed })
    );
    if failed.is_empty() {
        return Ok(());
    }
    Err(Failure::found(
        json!({ "code": "game_rule", "path": "$.pairs", "got": failed }).to_string(),
    ))
}
