//! The browser smoke checks: every `web/scripts/*-smoke.mjs`, found by
//! name, each opening with a `/**` comment
//! that says what it checks. They drive a running development server
//! (`DEMO_URL`, Vite's default otherwise; see `web/scripts/harness.mjs`).

use std::{fs, path::Path, process::Command};

/// Every smoke check: its name (`terrain` for `terrain-smoke.mjs`) and the
/// first line of what it checks. A check without its opening comment is an
/// error.
fn smokes(root: &Path) -> Result<Vec<(String, String)>, String> {
    let directory = root.join("web/scripts");
    let entries = fs::read_dir(&directory)
        .map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
    let mut smokes = Vec::new();
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
        let source = fs::read_to_string(directory.join(file))
            .map_err(|error| format!("cannot read {file}: {error}"))?;
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
        smokes.push((name.to_owned(), summary));
    }
    smokes.sort();
    Ok(smokes)
}

/// `smoke --list` lists the checks; `smoke NAME...` runs those in order and
/// `smoke --all` every one, stopping at the first failure.
pub fn smoke(root: &Path, options: &[String]) -> Result<(), String> {
    let smokes = smokes(root)?;
    let names: Vec<&str> = match options {
        [] => return Err("usage: smoke --list | --all | NAME...".to_owned()),
        [option] if option == "--list" => {
            for (name, summary) in &smokes {
                println!("{name:24} {summary}");
            }
            return Ok(());
        }
        [option] if option == "--all" => smokes.iter().map(|(name, _)| name.as_str()).collect(),
        names => names.iter().map(String::as_str).collect(),
    };
    for name in &names {
        if !smokes.iter().any(|(smoke, _)| smoke == name) {
            return Err(format!(
                "no smoke check named {name}; `cargo xtask smoke --list` lists them"
            ));
        }
    }
    for name in names {
        let script = format!("scripts/{name}-smoke.mjs");
        println!("smoke {name}");
        let status = Command::new("node")
            .arg(&script)
            .current_dir(root.join("web"))
            .status()
            .map_err(|error| format!("cannot run node {script}: {error}"))?;
        if !status.success() {
            return Err(format!("smoke check {name} failed"));
        }
    }
    Ok(())
}
