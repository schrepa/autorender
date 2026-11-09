use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime};

use serde_json::{Value, json};

use crate::checkouts::checkout;
use crate::measure::{commit_id, file_hash, git, iso, output, read_json};
use crate::process::Project;

const ATTEMPTS: usize = 3;

fn holds(expect: &str, exit_code: i32) -> bool {
    match expect {
        "fail" => exit_code == 3,
        _ => exit_code == 0,
    }
}

fn build(
    root: &Path,
    source: &Path,
    sha: &str,
) -> Result<(Result<PathBuf, String>, Value), String> {
    let project = Project::read(source, crate::smoke::PLAYER)?;
    let quick = fs::read_to_string(source.join("Cargo.toml"))
        .is_ok_and(|manifest| manifest.contains("[profile.quick]"));
    let profile = if quick {
        crate::pipeline::QUICK
    } else {
        "release"
    };
    let target = root.join("target").join("measure").join("check-target");
    let started = Instant::now();
    let built = Command::new("cargo")
        .args(["build", "-p", &project.package, "--bin", &project.native])
        .args(["--profile", profile, "--features", "dev", "--locked"])
        .env("CARGO_TARGET_DIR", &target)
        .env("ENGINE_COMMIT", &sha[..12])
        .current_dir(source)
        .status()
        .map_err(|error| format!("cannot run cargo: {error}"))
        .and_then(|status| {
            if status.success() {
                Ok(target.join(profile).join(project.program()))
            } else {
                Err(format!(
                    "building {} at {sha} failed ({status})",
                    project.package
                ))
            }
        });
    let artifact_hash = match &built {
        Ok(program) => Some(file_hash(program)?),
        Err(_) => None,
    };
    let identity = json!({
        "sha": sha,
        "profile": profile,
        "target": "native",
        "features": ["dev"],
        "toolchain": {
            "rustc": output(source, "rustc", &["-V"])?,
            "cargo": output(source, "cargo", &["-V"])?,
        },
        "env": {"RUSTFLAGS": std::env::var("RUSTFLAGS").unwrap_or_default()},
        "artifact_hash": artifact_hash,
        "build_ms": started.elapsed().as_secs_f64() * 1000.0,
    });
    Ok((built, identity))
}

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
        limit_ms(check)?;
        by_commit.entry(sha).or_default().push(check);
    }
    let (mut entries, mut incomplete) = (Vec::new(), false);
    for (sha, checks) in by_commit {
        let source = checkout(root, sha, None)?;
        eprintln!("{}", json!({"event": "building", "commit": sha}));
        let (program, identity) = build(root, &source, sha)?;
        for check in checks {
            let name = check["name"].as_str().ok_or("a check has a name")?;
            let expect = check["expect"].as_str().unwrap_or("pass");
            let limit = limit_ms(check)?;
            let mut attempts = Vec::new();
            while attempts.len() < ATTEMPTS {
                if !attempts.is_empty()
                    && deadline.is_some_and(|deadline| SystemTime::now() > deadline)
                {
                    incomplete = true;
                    break;
                }
                let started_at = iso(SystemTime::now());
                let started = Instant::now();
                let attempt_log =
                    folder.join(format!("{name}-{}-{}.log", &sha[..12], attempts.len() + 1));
                let exit_code = match &program {
                    Ok(program) => attempt(program, &source, (name, limit), &attempt_log)?,
                    Err(error) => {
                        fs::write(&attempt_log, format!("{error}\n"))
                            .map_err(|error| error.to_string())?;
                        -1
                    }
                };
                let record = json!({
                    "event": "check_attempted",
                    "check": name,
                    "commit": &sha[..12],
                    "attempt": attempts.len() + 1,
                    "exit_code": exit_code,
                });
                eprintln!("{record}");
                attempts.push(json!({
                    "started_at": started_at,
                    "duration_ms": started.elapsed().as_secs_f64() * 1000.0,
                    "exit_code": exit_code,
                    "log": attempt_log.display().to_string(),
                }));
                if holds(expect, exit_code) || program.is_err() {
                    break;
                }
            }
            entries.push(json!({
                "name": name,
                "sha": sha,
                "expect": expect,
                "build": identity,
                "attempts": attempts,
            }));
        }
    }
    crate::checkouts::prune(root);
    Ok((entries, incomplete))
}

fn limit_ms(check: &Value) -> Result<u64, String> {
    check["limit_ms"]
        .as_u64()
        .filter(|limit| *limit > 0)
        .ok_or_else(|| format!("the check {} has no wall-clock limit_ms", check["name"]))
}

fn attempt(
    program: &Path,
    source: &Path,
    (name, limit): (&str, u64),
    log: &Path,
) -> Result<i32, String> {
    let ran = Command::new(program)
        .args(["--smoke", name, "--limit-ms", &limit.to_string()])
        .current_dir(source)
        .output()
        .map_err(|error| format!("cannot run {}: {error}", program.display()))?;
    let stdout = String::from_utf8_lossy(&ran.stdout);
    let stderr = String::from_utf8_lossy(&ran.stderr);
    fs::write(log, format!("{stdout}\n{stderr}")).map_err(|error| error.to_string())?;
    Ok(judged(ran.status.code(), &stdout))
}

fn judged(code: Option<i32>, stdout: &str) -> i32 {
    let records: Vec<Value> = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|record| record["smoke"].is_string())
        .collect();
    let failed = records.iter().any(|record| record["ok"] != true);
    match code {
        Some(0) if records.is_empty() => -1,
        Some(0) if failed => 3,
        Some(code) => code,
        None => -1,
    }
}

pub fn check_task(root: &Path, arguments: &[String]) -> Result<(), String> {
    let usage = "usage: cargo xtask checks --task TASK.json --profile PROFILE.json --out REPORT.json [--formats FORMATS.json]";
    let mut found: BTreeMap<&str, PathBuf> = BTreeMap::new();
    let mut given = arguments.iter();
    while let Some(name) = given.next() {
        let key = match name.as_str() {
            "--task" | "--profile" | "--out" | "--formats" => &name[2..],
            _ => return Err(format!("unknown option {name}\n{usage}")),
        };
        let value = given
            .next()
            .ok_or_else(|| format!("{name} takes a path\n{usage}"))?;
        found.insert(key, PathBuf::from(value));
    }
    let [Some(task), Some(profile), Some(out)] =
        ["task", "profile", "out"].map(|key| found.remove(key))
    else {
        return Err(usage.into());
    };
    let task = read_json(&task)?;
    let spec = if task["spec"].is_object() {
        &task["spec"]
    } else {
        &task
    };
    let id = task["id"].as_u64().ok_or("the task has no id")?;
    let profile_start = read_json(&profile)?;
    let tools_sha = git(root, &["rev-parse", "HEAD"])?;
    if let Some(wanted) = spec["tools_sha"]
        .as_str()
        .filter(|wanted| *wanted != tools_sha)
    {
        return Err(format!(
            "the tools come from {wanted}; this checkout is {tools_sha} (git checkout --detach {wanted})"
        ));
    }
    let setup = match &spec["setup"] {
        Value::String(id) => id.clone(),
        setup => setup["id"].as_str().unwrap_or("unknown").to_owned(),
    };
    let deadline = spec["deadline"]
        .as_str()
        .and_then(crate::measure::parse_iso);
    let folder = root
        .join("target")
        .join("measure")
        .join("checks")
        .join(id.to_string());
    let checks = spec["checks"].as_array().cloned().unwrap_or_default();
    let (checks, incomplete) = run(root, &checks, &folder, deadline)?;
    let report = json!({
        "kind": "check",
        "task": id,
        "tools_sha": tools_sha,
        "profile_start": profile_start,
        "profile_end": read_json(&profile)?,
        "setup": setup,
        "sha": spec["sha"],
        "main_sha": spec["main_sha"],
        "checks": checks,
        "incomplete": incomplete,
    });
    fs::write(
        &out,
        serde_json::to_string_pretty(&report).expect("report_serializes") + "\n",
    )
    .map_err(|error| format!("cannot write {}: {error}", out.display()))?;
    for check in report["checks"].as_array().into_iter().flatten() {
        let codes: Vec<&Value> = check["attempts"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|attempt| &attempt["exit_code"])
            .collect();
        let record = json!({
            "event": "check_result",
            "check": check["name"],
            "expect": check["expect"],
            "exit_codes": codes,
        });
        println!("{record}");
    }
    println!(
        "{}",
        json!({"event": "written", "file": out.display().to_string()})
    );
    match found.remove("formats") {
        Some(formats) => crate::measure::check_format(&report, &formats, "check"),
        None => Ok(()),
    }
}
