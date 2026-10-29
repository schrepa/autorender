use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

use crate::Failure;
use crate::heavy::{HELD, HeavyRun};
use crate::pipeline::QUICK;
use crate::process::{Project, engine_commit, project_option, run};
use crate::serve::Server;

pub const PLAYER: &str = "crates/engine_player";

const USAGE: &str = "usage: cargo xtask smoke [NAME...] [--project DIR] [--browser] [--native-profile PROFILE] | [--project DIR] --list";

const BROWSER_LIMIT_MS: &str = "1800000";

struct Options {
    project: String,
    names: Vec<String>,
    browser: bool,
    list: bool,
    profile: String,
}

fn options(arguments: &[String]) -> Result<Options, String> {
    let (project, rest) = project_option(arguments)?;
    let mut options = Options {
        project: project.unwrap_or_else(|| PLAYER.to_owned()),
        names: Vec::new(),
        browser: false,
        list: false,
        profile: QUICK.to_owned(),
    };
    let mut given = rest.into_iter();
    while let Some(option) = given.next() {
        match option.as_str() {
            "--browser" => options.browser = true,
            "--list" => options.list = true,
            "--native-profile" => {
                options.profile = given
                    .next()
                    .filter(|profile| !profile.starts_with("--"))
                    .ok_or(USAGE)?;
            }
            flag if flag.starts_with("--") => return Err(format!("{flag}\n{USAGE}")),
            _ => options.names.push(option),
        }
    }
    if options.list && (options.browser || !options.names.is_empty()) {
        return Err(USAGE.to_owned());
    }
    Ok(options)
}

pub fn smoke(root: &Path, arguments: &[String]) -> Result<(), Failure> {
    let options = options(arguments)?;
    let project = Project::read(root, &options.project)?;
    if options.list {
        let program = build(root, &project, &options.profile)?;
        return Ok(run(root, &program, &["--smoke", engine::smokes::LIST])?);
    }
    let names = if options.names.is_empty() {
        engine::smokes::ALL.to_owned()
    } else {
        options.names.join(",")
    };
    let _heavy = HeavyRun::hold(&format!("cargo xtask smoke {names}"))?;
    let records = if options.browser {
        browser(root, &project, &names)?
    } else {
        let program = build(root, &project, &options.profile)?;
        let mut command = Command::new(&program);
        command
            .args(["--smoke", &names])
            .envs([engine_commit()])
            .current_dir(root);
        collect(command, &program)?
    };
    judged(&records)
}

fn build(root: &Path, project: &Project, profile: &str) -> Result<String, String> {
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

fn browser(root: &Path, project: &Project, names: &str) -> Result<Vec<Value>, Failure> {
    let page = project
        .client()
        .and_then(|client| client.page.clone())
        .ok_or_else(|| format!("{}: page", project.package))?;
    let server = Server::page(root, &root.join("target").join("smoke").join("server.log"))?;
    let url = format!("{}/{page}?smoke={names}", server.url);
    let mut command = Command::new("node");
    command
        .args(["scripts/smoke-browser.mjs", &url, BROWSER_LIMIT_MS])
        .current_dir(root.join("web"))
        .env(HELD.0, HELD.1);
    let records = collect(command, "scripts/smoke-browser.mjs");
    drop(server);
    records
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
    let named = |wanted: fn(&Value) -> bool| -> Vec<&str> {
        records
            .iter()
            .filter(|record| wanted(record))
            .filter_map(|record| record["smoke"].as_str())
            .collect()
    };
    let failed = named(|record| record["ok"] != true);
    let skipped = named(|record| record["skipped"] == true);
    println!(
        "{}",
        json!({
            "smokes": records.len(),
            "failed": failed,
            "skipped": skipped,
        })
    );
    if failed.is_empty() {
        return Ok(());
    }
    Err(Failure::found(
        json!({ "code": "game_rule", "path": "$.smokes", "got": failed }).to_string(),
    ))
}
