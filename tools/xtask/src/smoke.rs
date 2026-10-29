use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::{fs, process::Command};

use serde_json::{Value, json};

use crate::Failure;
use crate::heavy::{HELD, HeavyRun};
use crate::serve::Server;

pub fn scripts(root: &Path) -> Result<Vec<(String, String)>, String> {
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
        scripts.push((name.to_owned(), source));
    }
    scripts.sort();
    Ok(scripts)
}

pub fn checks(root: &Path, project: &str) -> Result<Vec<(String, PathBuf)>, String> {
    let directory = root.join(project).join("checks");
    let entries = match fs::read_dir(&directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("cannot read {}: {error}", directory.display())),
    };
    let mut checks = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|error| format!("cannot read a check entry: {error}"))?
            .path();
        let name = path
            .file_name()
            .and_then(|file| file.to_str())
            .and_then(|file| file.strip_suffix(".mjs"));
        if let Some(name) = name {
            checks.push((name.to_owned(), path.clone()));
        }
    }
    checks.sort();
    Ok(checks)
}

fn sent_folder(root: &Path) -> PathBuf {
    root.join("target").join("smoke-coverage")
}

fn smokes(root: &Path, project: Option<&str>) -> Result<Vec<(String, PathBuf)>, String> {
    match project {
        Some(project) => checks(root, project),
        None => Ok(scripts(root)?
            .into_iter()
            .map(|(name, _)| {
                let path = root.join("web/scripts").join(format!("{name}-smoke.mjs"));
                (name, path)
            })
            .collect()),
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub fn smoke(root: &Path, options: &[String]) -> Result<(), Failure> {
    let usage = "usage: smoke [--project DIR] --list | [--project DIR] [--serve] [--keep-going] [--native-profile PROFILE] (--all | NAME...)";
    let (project, options) = crate::process::project_option(options)?;
    let smokes = smokes(root, project.as_deref())?;
    let (mut serve, mut keep_going, mut all) = (false, false, false);
    let mut native_profile = None;
    let mut names: Vec<&str> = Vec::new();
    let mut given = options.iter();
    while let Some(option) = given.next() {
        match option.as_str() {
            "--list" if options.len() == 1 => {
                for (name, _) in &smokes {
                    println!("{name}");
                }
                return Ok(());
            }
            "--serve" => serve = true,
            "--keep-going" => keep_going = true,
            "--all" => all = true,
            "--native-profile" => {
                native_profile = Some(
                    given
                        .next()
                        .filter(|profile| !profile.starts_with("--"))
                        .ok_or(usage)?,
                );
            }
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
    let program = std::env::current_exe()
        .map_err(|error| format!("cannot name the running program: {error}"))?;
    let _heavy = HeavyRun::hold(&format!("cargo xtask smoke {}", names.join(" ")))?;
    let mut results: Vec<(&str, Option<i32>, Duration)> = Vec::new();
    for name in &names {
        let script = smokes
            .iter()
            .find(|(smoke, _)| smoke == name)
            .map(|(_, path)| path.display().to_string())
            .unwrap_or_default();
        println!("smoke {name}");
        let started = Instant::now();
        let mut command = Command::new("node");
        let sent = sent_folder(root).join(name);
        if let Err(error) = fs::remove_dir_all(&sent)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            return Err(format!("cannot clear {}: {error}", sent.display()).into());
        }
        command
            .arg(&script)
            .current_dir(root.join("web"))
            .env("ENGINE_WORKSPACE", root)
            .env("ENGINE_SMOKE_SENT", &sent)
            .env("ENGINE_XTASK_BIN", &program)
            .env(HELD.0, HELD.1);
        if let Some(server) = &server {
            command.env("DEMO_URL", &server.url);
        }
        if let Some(profile) = native_profile {
            command.env("ENGINE_NATIVE_PROFILE", profile);
        }
        let status = command
            .status()
            .map_err(|error| format!("cannot run node {script}: {error}"))?;
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
    let passed: Vec<&str> = results
        .iter()
        .filter(|(_, code, _)| *code == Some(0))
        .map(|(name, _, _)| *name)
        .collect();
    let everything = all && passed.len() == names.len();
    if let Err(error) = report_sent(root, &passed, everything) {
        println!("No record of what the checks sent: {error}");
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

#[derive(Default)]
struct Sent {
    ops: BTreeSet<String>,
    sections: BTreeSet<String>,
    inspectors: BTreeSet<String>,
}

fn sent(folder: &Path) -> Result<Sent, String> {
    let mut sent = Sent::default();
    let entries = match fs::read_dir(folder) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(sent),
        Err(error) => return Err(format!("cannot read {}: {error}", folder.display())),
    };
    for entry in entries {
        let path = entry.map_err(|error| error.to_string())?.path();
        let record = crate::measure::read_json(&path)?;
        for (field, into) in [
            ("ops", &mut sent.ops),
            ("sections", &mut sent.sections),
            ("inspectors", &mut sent.inspectors),
        ] {
            into.extend(
                record[field]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .map(str::to_owned),
            );
        }
    }
    Ok(sent)
}

fn report_sent(root: &Path, passed: &[&str], everything: bool) -> Result<(), String> {
    let engine = crate::bricks::standard_engine();
    let development = engine_dev::schemas::generated(engine.app.world())?;
    let offered: Value = serde_json::from_str(&crate::coverage::coverage(root, &development)?)
        .map_err(|error| format!("the coverage file is not JSON: {error}"))?;
    let section_of: BTreeMap<&str, &str> = crate::bricks::environment_sections(engine.app.world())
        .into_iter()
        .map(|(key, settings)| (settings, key))
        .collect();
    let lists = [
        ("ops", "op"),
        ("sections", "section"),
        ("inspectors", "inspector"),
    ];
    let mut checks = serde_json::Map::new();
    let mut unconfirmed = Vec::new();
    let mut union: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    for name in passed {
        let mut sent = sent(&sent_folder(root).join(name))?;
        let inspected_sections: Vec<String> = sent
            .inspectors
            .iter()
            .filter_map(|inspector| {
                section_of
                    .get(inspector.as_str())
                    .map(|key| (*key).to_owned())
            })
            .collect();
        sent.sections.extend(inspected_sections);
        let mut claimed_unsent = Vec::new();
        for (list, key) in lists {
            let exercised = match list {
                "ops" => &sent.ops,
                "sections" => &sent.sections,
                _ => &sent.inspectors,
            };
            union
                .entry(list)
                .or_default()
                .extend(exercised.iter().cloned());
            for entry in offered[list].as_array().into_iter().flatten() {
                let item = entry[key].as_str().unwrap_or_default();
                let claims = entry["smokes"]
                    .as_array()
                    .is_some_and(|smokes| smokes.iter().any(|smoke| smoke == name));
                if claims && !exercised.contains(item) {
                    claimed_unsent.push(format!("{key} {item}"));
                }
            }
        }
        if !claimed_unsent.is_empty() {
            unconfirmed.push(format!("{name}: {}", claimed_unsent.join(", ")));
        }
        checks.insert(
            (*name).to_owned(),
            json!({
                "ops": sent.ops,
                "sections": sent.sections,
                "inspectors": sent.inspectors,
                "claimed_not_sent": claimed_unsent,
            }),
        );
    }
    let mut counts = Vec::new();
    let mut never = Vec::new();
    for (list, key) in lists {
        let items: Vec<&str> = offered[list]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry[key].as_str())
            .collect();
        let exercised = union.get(list);
        let reached = items
            .iter()
            .filter(|item| exercised.is_some_and(|set| set.contains(**item)))
            .count();
        counts.push(format!("{reached} of {} {list}", items.len()));
        if everything {
            never.extend(
                items
                    .iter()
                    .filter(|item| !exercised.is_some_and(|set| set.contains(**item)))
                    .map(|item| format!("{key} {item}")),
            );
        }
    }
    let report = json!({
        "checks": checks,
        "sent_by_none": if everything { json!(never) } else { Value::Null },
    });
    let path = sent_folder(root).join("report.json");
    fs::create_dir_all(sent_folder(root)).map_err(|error| error.to_string())?;
    fs::write(
        &path,
        serde_json::to_string_pretty(&report).expect("reports serialize") + "\n",
    )
    .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    if passed.is_empty() {
        return Ok(());
    }
    println!(
        "\nThe checks that passed sent {} ({}).",
        counts.join(", "),
        path.display()
    );
    if !unconfirmed.is_empty() {
        println!("Named in a check's code but not sent when it ran:");
        for line in &unconfirmed {
            println!("  {line}");
        }
    }
    if everything && !never.is_empty() {
        println!("Sent by no check: {}.", never.join(", "));
    }
    Ok(())
}
