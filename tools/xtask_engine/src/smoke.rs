use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use serde_json::{Value, json};
use xtask::serve::Server;

use crate::Failure;
use crate::pipeline::build_native;
use crate::process::{Project, project_option, run};

pub const PLAYER: &str = "crates/engine_player";

const USAGE: &str =
    "usage: cargo xtask smoke [NAME...] --limit-ms MS [--project DIR] [--browser] | [--project DIR] --list";

const BROWSER_RUNNER: &str = "scripts/smoke.mjs";

struct Options {
    project: String,
    names: Vec<String>,
    list: bool,
    limit_ms: Option<String>,
    browser: bool,
}

fn options(arguments: &[String]) -> Result<Options, String> {
    let (project, rest) = project_option(arguments)?;
    let mut options = Options {
        project: project.unwrap_or_else(|| PLAYER.to_owned()),
        names: Vec::new(),
        list: false,
        limit_ms: None,
        browser: false,
    };
    let mut given = rest.into_iter();
    while let Some(option) = given.next() {
        match option.as_str() {
            "--list" => options.list = true,
            "--browser" => options.browser = true,
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
    if options.list && (!options.names.is_empty() || options.limit_ms.is_some() || options.browser)
    {
        return Err(USAGE.to_owned());
    }
    Ok(options)
}

pub fn smoke(root: &Path, arguments: &[String]) -> Result<(), Failure> {
    let options = options(arguments)?;
    let project = Project::read(root, &options.project)?;
    let program = build_native(root, &project)?;
    if options.list {
        return Ok(run(root, &program, &["--smoke", engine::smokes::LIST])?);
    }
    let names = if options.names.is_empty() {
        engine::smokes::ALL.to_owned()
    } else {
        options.names.join(",")
    };
    let limit = options.limit_ms.ok_or_else(|| USAGE.to_owned())?;
    if options.browser {
        return browser(root, &project, &program, (&names, &limit));
    }
    let mut command = Command::new(&program);
    command
        .args(["--smoke", &names, "--limit-ms", &limit])
        .current_dir(root);
    judged(&collect(command, &program)?)
}

fn browser(
    root: &Path,
    project: &Project,
    program: &str,
    (names, limit): (&str, &str),
) -> Result<(), Failure> {
    let page = project
        .client()
        .and_then(|client| client.page.clone())
        .ok_or_else(|| format!("{}: page", project.package))?;
    let limit_ms = limit.parse::<u64>().map_err(|_| USAGE.to_owned())?;
    let log = root
        .join("target")
        .join("smoke")
        .join(format!("{}-browser", project.package))
        .join("server.log");
    let server = Server::page(root, &log, Duration::from_millis(limit_ms))?;
    let mut command = Command::new("node");
    command
        .args([BROWSER_RUNNER, "--native", program])
        .args(["--url", &format!("{}/{page}", server.url)])
        .args(["--smoke", names, "--limit-ms", limit])
        .current_dir(root.join("web"));
    let records = collect(command, BROWSER_RUNNER);
    drop(server);
    judged(&records?)
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
