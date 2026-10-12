//! The browser smoke checks: every `web/scripts/*-smoke.mjs`, found by
//! name, each opening with a `/**` comment
//! that says what it checks. They drive a running development server
//! (`DEMO_URL`, Vite's default otherwise; see `web/scripts/harness.mjs`), or
//! with `--serve` one of this checkout that the run starts and stops, and
//! build, run and write in this checkout (`ENGINE_WORKSPACE`). A run holds
//! the machine's heavy-run lock throughout ([`crate::heavy`]); each check
//! exits 0, 3 for a failed assertion or 1 for anything else that stopped
//! it, and so does the run.

use std::time::{Duration, Instant};
use std::{fs, path::Path, process::Command};

use crate::Failure;
use crate::heavy::{HELD, HeavyRun};
use crate::serve::Server;

/// Every smoke check's name (`terrain` for `terrain-smoke.mjs`), file name
/// and source, by name. Local scratch scripts (`zz-`, which git ignores)
/// are not checks.
pub fn scripts(root: &Path) -> Result<Vec<(String, String, String)>, String> {
    let directory = root.join("web/scripts");
    let entries = fs::read_dir(&directory)
        .map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
    let mut scripts = Vec::new();
    for entry in entries {
        let file = entry
            .map_err(|error| format!("cannot read a script entry: {error}"))?
            .file_name();
        let Some(file) = file.to_str() else {
            continue;
        };
        let Some(name) = file.strip_suffix("-smoke.mjs") else {
            continue;
        };
        if name.starts_with("zz-") {
            continue;
        }
        let source = fs::read_to_string(directory.join(file))
            .map_err(|error| format!("cannot read {file}: {error}"))?;
        scripts.push((name.to_owned(), file.to_owned(), source));
    }
    scripts.sort();
    Ok(scripts)
}

/// Every smoke check: its name and the first line of what it checks. A
/// check without its opening comment is an error.
fn smokes(root: &Path) -> Result<Vec<(String, String)>, String> {
    let mut smokes = Vec::new();
    for (name, file, source) in scripts(root)? {
        let summary = source
            .trim_start()
            .strip_prefix("/**")
            .map(|comment| {
                comment
                    .trim_start_matches(['*', ' ', '\r', '\n'])
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .trim_end_matches("*/")
                    .trim()
                    .to_owned()
            })
            .filter(|summary| !summary.is_empty())
            .ok_or_else(|| format!("{file} must open with a /** comment saying what it checks"))?;
        smokes.push((name, summary));
    }
    Ok(smokes)
}

/// `smoke --list` lists the checks; `smoke NAME...` runs those in order and
/// `smoke --all` every one, stopping at the first failure unless
/// `--keep-going`, against the server `DEMO_URL` names or, with `--serve`,
/// one of this checkout. Fails with exit code 3 when a check's assertion
/// failed, 1 when one stopped otherwise.
pub fn smoke(root: &Path, options: &[String]) -> Result<(), Failure> {
    let usage = "usage: smoke --list | [--serve] [--keep-going] (--all | NAME...)";
    let smokes = smokes(root)?;
    let (mut serve, mut keep_going, mut all) = (false, false, false);
    let mut names: Vec<&str> = Vec::new();
    for option in options {
        match option.as_str() {
            "--list" if options.len() == 1 => {
                for (name, summary) in &smokes {
                    println!("{name:24} {summary}");
                }
                return Ok(());
            }
            "--serve" => serve = true,
            "--keep-going" => keep_going = true,
            "--all" => all = true,
            name if !name.starts_with("--") => names.push(name),
            _ => {
                return Err(format!(
                    "unknown option {option}
{usage}"
                )
                .into());
            }
        }
    }
    if all == !names.is_empty() {
        return Err(usage.into());
    }
    if all {
        names = smokes.iter().map(|(name, _)| name.as_str()).collect();
    }
    for name in &names {
        if !smokes.iter().any(|(smoke, _)| smoke == name) {
            return Err(format!(
                "no smoke check named {name}; `cargo xtask smoke --list` lists them"
            )
            .into());
        }
    }
    let server = if serve {
        Some(Server::page(
            root,
            &root.join("target").join("smoke").join("server.log"),
        )?)
    } else {
        None
    };
    let _heavy = HeavyRun::hold(&format!("cargo xtask smoke {}", names.join(" ")))?;
    let mut results: Vec<(&str, Option<i32>, Duration)> = Vec::new();
    for name in &names {
        let script = format!("scripts/{name}-smoke.mjs");
        println!("smoke {name}");
        let started = Instant::now();
        let mut command = Command::new("node");
        command
            .arg(&script)
            .current_dir(root.join("web"))
            .env("ENGINE_WORKSPACE", root)
            .env(HELD.0, HELD.1);
        if let Some(server) = &server {
            command.env("DEMO_URL", &server.url);
        }
        let status = command
            .status()
            .map_err(|error| format!("cannot run node {script}: {error}"))?;
        // A check that ended without an exit code (a signal) stopped.
        let code = status.code().unwrap_or(1);
        results.push((name, Some(code), started.elapsed()));
        if code != 0 && !keep_going {
            break;
        }
    }
    drop(server);
    let ran = results.len();
    results.extend(
        names[ran..]
            .iter()
            .map(|name| (*name, None, Duration::ZERO)),
    );
    let failed: Vec<(&str, i32)> = results
        .iter()
        .filter_map(|&(name, code, _)| code.filter(|code| *code != 0).map(|code| (name, code)))
        .collect();
    if results.len() > 1 || !failed.is_empty() {
        println!(
            "
{:24} {:>6} {:>9}",
            "smoke", "exit", "seconds"
        );
        for (name, code, took) in &results {
            match code {
                Some(code) => println!("{name:24} {code:>6} {:>9.1}", took.as_secs_f64()),
                None => println!("{name:24} {:>6} {:>9}", "-", "not run"),
            }
        }
    }
    if failed.is_empty() {
        return Ok(());
    }
    let listed: Vec<String> = failed
        .iter()
        .map(|(name, code)| format!("{name} (exit {code})"))
        .collect();
    let message = format!("smoke checks failed: {}", listed.join(", "));
    if failed.iter().any(|(_, code)| *code == 3) {
        Err(Failure::found(message))
    } else {
        Err(message.into())
    }
}
