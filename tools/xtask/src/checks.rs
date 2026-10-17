//! Local revision checks: smoke scripts (`web/scripts/NAME.mjs`) run
//! against a development server of one commit, with the scripts of the
//! same or another commit, every attempt kept. The scripts build, run and
//! write in the served commit's checkout (`ENGINE_WORKSPACE`), so another
//! commit's checks test this one's programs and files; the checkout holds
//! the files its pages read that git does not (`cargo xtask bench assets`),
//! as a measured commit's does. A check holds when its
//! first attempt does what it must (exit 0, or 3 for an assertion where it
//! must fail); one that does not runs again, up to `ATTEMPTS` in all, so a
//! failure followed by a pass shows as flaky. `cargo xtask checks` runs a
//! check task; measure tasks run theirs after their workloads.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime};

use serde_json::{Value, json};

use crate::checkouts::checkout;
use crate::measure::{commit_id, git, iso, output, read_json};
use crate::serve::{Server, free_port};

/// Attempts a check gets in all.
const ATTEMPTS: usize = 3;
/// How long a development server may take to build and start.
const SERVER_START: Duration = Duration::from_secs(30 * 60);

/// What a check must do: exit 0, or 3 (a failed assertion).
fn holds(expect: &str, exit_code: i32) -> bool {
    match expect {
        "fail" => exit_code == 3,
        _ => exit_code == 0,
    }
}

/// A development server of one commit, and what it serves.
struct Served {
    server: Server,
    build: Value,
}

/// Starts `cargo xtask dev` in `source` on a free port and waits until it
/// serves: its build of the page logged into `log`.
fn start_server(source: &Path, sha: &str, log: &Path, limit: Duration) -> Result<Served, String> {
    let port = free_port()?;
    let mut command = Command::new("cargo");
    command
        .args(["xtask", "dev", "--port", &port.to_string()])
        .env("ENGINE_TOOL_WORKSPACE", source)
        .current_dir(source);
    let started = Instant::now();
    let server = Server::start(command, port, log, sha, limit)?;
    let build = web_build(source, sha, started.elapsed())?;
    Ok(Served { server, build })
}

/// What a development server serves: the commit's page and its Wasm
/// clients with their development tools, built with the profile and
/// features its `web/pkg-dev/build.json` names (release and the development
/// tools for commits from before it was written).
fn web_build(source: &Path, sha: &str, took: Duration) -> Result<Value, String> {
    let identity = read_json(&source.join("web/pkg-dev/build.json")).unwrap_or_default();
    let profile = identity["profile"].as_str().unwrap_or("release");
    let features = match &identity["features"] {
        Value::Array(features) => Value::Array(features.clone()),
        _ => json!(["engine/dev"]),
    };
    let manifest = xtask::files::manifest(&source.join("web/pkg-dev"))?;
    let files: Vec<_> = manifest
        .as_array()
        .ok_or("binding manifest must be an array")?
        .iter()
        .filter(|file| file["path"] != "build.json")
        .cloned()
        .collect();
    let artifact_hash = xtask::files::digest(
        serde_json::to_string(&files)
            .map_err(|e| e.to_string())?
            .as_bytes(),
    );
    let wasm_bindgen = output(source, "wasm-bindgen", &["--version"]).unwrap_or_default();
    Ok(json!({
        "sha": sha,
        "profile": profile,
        "target": "web-wasm",
        "features": features,
        "toolchain": {
            "rustc": output(source, "rustc", &["-V"])?,
            "cargo": output(source, "cargo", &["-V"])?,
            "node": output(source, "node", &["--version"])?,
            "wasm-bindgen": wasm_bindgen,
        },
        "env": {"RUSTFLAGS": std::env::var("RUSTFLAGS").unwrap_or_default()},
        "artifact_hash": artifact_hash,
        "build_ms": took.as_secs_f64() * 1000.0,
    }))
}

/// A build that did not finish: its commit, the profile its development
/// server builds with (release before the quick profile existed), and as
/// its artifact the log that says why.
fn failed_build(source: &Path, sha: &str, log: &Path) -> Result<Value, String> {
    let quick = fs::read_to_string(source.join("Cargo.toml"))
        .is_ok_and(|manifest| manifest.contains("[profile.quick]"));
    Ok(json!({
        "sha": sha,
        "profile": if quick { crate::pipeline::QUICK } else { "release" },
        "target": "web-wasm",
        "features": ["engine/dev"],
        "toolchain": {
            "rustc": output(source, "rustc", &["-V"])?,
            "cargo": output(source, "cargo", &["-V"])?,
        },
        "env": {"RUSTFLAGS": std::env::var("RUSTFLAGS").unwrap_or_default()},
        "artifact_hash": crate::measure::file_hash(log)?,
        "build_ms": 0,
    }))
}

/// The web folder whose scripts run: `scripts_sha`'s, its dependencies
/// installed, or the served commit's own.
fn scripts_folder(
    root: &Path,
    source: &Path,
    scripts_sha: Option<&str>,
) -> Result<PathBuf, String> {
    let Some(sha) = scripts_sha else {
        return Ok(source.join("web"));
    };
    let scripts = checkout(root, sha, None)?;
    crate::pipeline::install_web_dependencies(&scripts)?;
    Ok(scripts.join("web"))
}

/// Runs `checks` (the task's: name, sha, scripts_sha, expect), each
/// commit's server started once, into `folder`; stops adding attempts at
/// `deadline`. The checks' entries, and whether the deadline cut any short.
pub fn run(
    root: &Path,
    checks: &[Value],
    folder: &Path,
    deadline: Option<SystemTime>,
) -> Result<(Vec<Value>, bool), String> {
    fs::create_dir_all(folder).map_err(|error| error.to_string())?;
    let mut by_commit: BTreeMap<&str, Vec<&Value>> = BTreeMap::new();
    for check in checks {
        let sha = commit_id(check["sha"].as_str(), "a check's sha")?;
        if !check["scripts_sha"].is_null() {
            commit_id(check["scripts_sha"].as_str(), "a check's scripts_sha")?;
        }
        by_commit.entry(sha).or_default().push(check);
    }
    let (mut entries, mut incomplete) = (Vec::new(), false);
    for (sha, checks) in by_commit {
        if deadline.is_some_and(|end| SystemTime::now() >= end) {
            incomplete = true;
            break;
        }
        let source = checkout(root, sha, None)?;
        // Its pages read the files git does not hold, which a measured
        // commit's checkout has too; without them they fail to load.
        if let Err(error) = crate::measure::generated(root, &source, sha) {
            eprintln!("The files {sha}'s pages read are missing: {error}");
        }
        eprintln!("Starting the development server of {sha}.");
        let log = folder.join(format!(
            "server-{}-{}.log",
            &sha[..12],
            xtask::files::unique()
        ));
        let limit = deadline
            .map(|end| end.duration_since(SystemTime::now()).unwrap_or_default())
            .unwrap_or(SERVER_START)
            .min(SERVER_START);
        let server = start_server(&source, sha, &log, limit);
        for check in checks {
            let name = check["name"].as_str().ok_or("a check has a name")?;
            if name.is_empty()
                || !name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
            {
                return Err("invalid check script name".into());
            }
            let expect = check["expect"].as_str().unwrap_or("pass");
            let scripts_sha = check["scripts_sha"].as_str();
            let mut attempts = Vec::new();
            let build = match &server {
                Ok(server) => server.build.clone(),
                Err(_) => failed_build(&source, sha, &log)?,
            };
            let web = match &server {
                Ok(_) => Some(scripts_folder(root, &source, scripts_sha)?),
                Err(_) => None,
            };
            while attempts.len() < ATTEMPTS {
                if deadline.is_some_and(|deadline| SystemTime::now() > deadline) {
                    incomplete = true;
                    break;
                }
                let started_at = iso(SystemTime::now());
                let started = Instant::now();
                let attempt_log = folder.join(format!(
                    "{name}-{}-{}-{}.log",
                    &sha[..12],
                    attempts.len() + 1,
                    xtask::files::unique()
                ));
                let (exit_code, measurements) = match (&server, &web) {
                    (Ok(served), Some(web)) => attempt(
                        web,
                        &source,
                        name,
                        &served.server.url,
                        &attempt_log,
                        deadline,
                    )?,
                    (Err(error), _) => {
                        fs::write(&attempt_log, format!("{error}\n"))
                            .map_err(|error| error.to_string())?;
                        (-1, Vec::new())
                    }
                    (Ok(_), None) => unreachable!("a served commit has its scripts"),
                };
                eprintln!(
                    "Check {name} on {}: attempt {} exited {exit_code}.",
                    &sha[..12],
                    attempts.len() + 1
                );
                attempts.push(json!({
                    "started_at": started_at,
                    "duration_ms": started.elapsed().as_secs_f64() * 1000.0,
                    "exit_code": exit_code,
                    "log": attempt_log.display().to_string(),
                    "measurements": measurements,
                }));
                xtask::files::atomic_json(
                    &attempt_log.with_extension("attempt.json"),
                    attempts.last().expect("recorded attempt"),
                )?;
                xtask::files::atomic_json(
                    &folder.join("checks.checkpoint.json"),
                    &json!({"completed": entries, "current": {"name": name, "sha": sha, "expect": expect, "build": build, "attempts": attempts}, "incomplete": true}),
                )?;
                // Nothing to run again without a server.
                if holds(expect, exit_code) || server.is_err() {
                    break;
                }
            }
            if attempts.is_empty() {
                incomplete = true;
                break;
            }
            entries.push(json!({
                "name": name,
                "sha": sha,
                "scripts_sha": scripts_sha,
                "expect": expect,
                "build": build,
                "attempts": attempts,
            }));
        }
        drop(server);
    }
    crate::checkouts::prune(root);
    Ok((entries, incomplete))
}

/// One attempt of check `name` from `web` against `url`, a server of the
/// checkout `workspace`, which the scripts build, run and write in: its exit
/// code and what it measured, its output into `log`.
fn attempt(
    web: &Path,
    workspace: &Path,
    name: &str,
    url: &str,
    log: &Path,
    deadline: Option<SystemTime>,
) -> Result<(i32, Vec<Value>), String> {
    let script = format!("scripts/{name}.mjs");
    if !web.join(&script).is_file() {
        fs::write(log, format!("{script} does not exist\n")).map_err(|error| error.to_string())?;
        return Ok((-1, Vec::new()));
    }
    let mut command = Command::new("node");
    command
        .arg(&script)
        .env("DEMO_URL", url)
        .env("ENGINE_WORKSPACE", workspace)
        .env("ENGINE_TOOL_WORKSPACE", workspace)
        .current_dir(web);
    let timeout = deadline
        .map(|end| end.duration_since(SystemTime::now()).unwrap_or_default())
        .unwrap_or(Duration::from_secs(30 * 60));
    let raw = log.with_extension("raw");
    let result = xtask::jobs::run(
        &mut command,
        &raw,
        &xtask::jobs::Limits {
            timeout,
            ..Default::default()
        },
    );
    let stdout = fs::read_to_string(raw.join("stdout.log")).unwrap_or_default();
    let stderr = fs::read_to_string(raw.join("stderr.log")).unwrap_or_default();
    let (code, diagnostic) = match result {
        Ok(output) => (output.status.code().unwrap_or(-1), String::new()),
        Err(error) => (-1, error),
    };
    fs::write(log, format!("{stdout}\n{stderr}\n{diagnostic}")).map_err(|e| e.to_string())?;
    Ok((code, measurements(&stdout)))
}

/// What a check measured about the image or the engine: the `measurements`
/// of the last line it printed, when that line is a JSON object, each
/// `{name, value, unit, note?}`; entries of another shape are left out.
/// Numbers for the user to read, never a limit the check passes or fails.
fn measurements(stdout: &str) -> Vec<Value> {
    let last = stdout
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty());
    let Some(Ok(Value::Object(last))) = last.map(serde_json::from_str::<Value>) else {
        return Vec::new();
    };
    let text = |entry: &Value, key: &str| {
        entry[key]
            .as_str()
            .filter(|text| !text.is_empty())
            .map(crate::measure::reason)
    };
    last.get("measurements")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let mut kept = json!({
                "name": text(entry, "name")?,
                "value": entry["value"].as_f64()?,
                "unit": text(entry, "unit")?,
            });
            if let Some(note) = text(entry, "note") {
                kept["note"] = note.into();
            }
            Some(kept)
        })
        .collect()
}

/// Execute a versioned local request, preserving every check attempt.
pub fn check_task(root: &Path, arguments: &[String]) -> Result<(), String> {
    let execution = xtask::execution::Request::read(arguments, "checks")?;
    if execution.previous.is_some() {
        return Err("check requests run fresh; continuation is for paired measurements".into());
    }
    let spec = &execution.spec;
    let deadline = spec["deadline"]
        .as_str()
        .and_then(crate::measure::parse_iso);
    let wanted = spec["checks"]
        .as_array()
        .cloned()
        .ok_or("checks must be an array")?;
    let mut checks = Vec::new();
    let mut incomplete = false;
    let mut report = json!({"kind": "check", "tools_sha": git(root, &["rev-parse", "HEAD"])? ,
        "tool_identity": xtask::execution::tool_identity(root)?,
        "profile_start": spec["environment"], "profile_end": spec["environment"],
        "setup": spec["setup"]["id"], "sha": spec["sha"], "main_sha": spec["main_sha"],
        "checks": [], "incomplete": true});
    execution.checkpoint("checks", &report)?;
    // Grouped execution keeps one server per served revision. Its per-attempt
    // journal is durable even if the process is interrupted before returning.
    let (completed, cut) = run(root, &wanted, &execution.folder, deadline)?;
    checks.extend(completed);
    incomplete |= cut;
    report["checks"] = json!(checks);
    report["incomplete"] = json!(incomplete);
    execution.checkpoint("complete", &report)?;
    for check in report["checks"].as_array().into_iter().flatten() {
        let codes: Vec<String> = check["attempts"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|attempt| attempt["exit_code"].to_string())
            .collect();
        println!(
            "{} ({}): exited {}",
            check["name"].as_str().unwrap_or_default(),
            check["expect"].as_str().unwrap_or_default(),
            codes.join(", then ")
        );
        for attempt in check["attempts"].as_array().into_iter().flatten() {
            for measured in attempt["measurements"].as_array().into_iter().flatten() {
                println!(
                    "  {}: {} {}",
                    measured["name"].as_str().unwrap_or_default(),
                    measured["value"],
                    measured["unit"].as_str().unwrap_or_default()
                );
            }
        }
    }
    println!("Wrote {}.", execution.out.display());
    Ok(())
}
