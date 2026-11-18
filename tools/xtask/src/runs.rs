use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use engine_command_line::{Command, CommandOption};
use serde_json::{Value, json};

use crate::Failure;

pub struct Run {
    pub name: String,
    pub folder: PathBuf,
    pub header: Value,
    pub entries: Vec<Value>,
}

pub fn commands() -> Vec<Command> {
    let name = || CommandOption::taking("NAME", "NAME").required();
    vec![
        Command::new("runs", []),
        Command::new("runs delete", [name()]),
        Command::new("runs causes", [name()]),
    ]
}

pub fn runs(root: &Path, arguments: &[String]) -> Result<(), Failure> {
    match arguments {
        [] => list(root),
        [verb, name] if verb == "delete" => delete(root, name),
        [verb, name] if verb == "causes" => causes(root, name),
        _ => Err(json!({
            "code": "invalid_arguments",
            "path": "runs",
            "got": arguments.join(" "),
            "expected": "runs | runs delete <name> | runs causes <name>",
        })
        .to_string()
        .into()),
    }
}

#[derive(Default)]
struct Totals {
    opened: Option<(String, String, u64)>,
    jobs: u64,
    ran_s: f64,
    bytes: u64,
    pipelines: u64,
    first_drawn: Option<u64>,
}

fn causes(root: &Path, name: &str) -> Result<(), Failure> {
    let run = read(root, name)?;
    let traced = run.header["instruments"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|instrument| instrument["name"] == "causes");
    let mut totals: BTreeMap<u64, Totals> = BTreeMap::new();
    totals.entry(0).or_default();
    let mut dropped = 0;
    for entry in &run.entries {
        dropped += entry["dropped"]["events"].as_u64().unwrap_or(0);
        for event in entry["events"].as_array().into_iter().flatten() {
            let total = totals
                .entry(event["cause"].as_u64().unwrap_or(0))
                .or_default();
            let frame = event["frame"].as_u64().unwrap_or(0);
            match event["kind"].as_str().unwrap_or_default() {
                "cause" => {
                    let text = |key: &str| event[key].as_str().unwrap_or_default().to_owned();
                    total.opened = Some((text("name"), text("what"), frame));
                }
                "job" => {
                    total.jobs += 1;
                    total.ran_s += event["ran_s"].as_f64().unwrap_or(0.0);
                }
                "write" => total.bytes += event["bytes"].as_u64().unwrap_or(0),
                "pipeline" if event["state"] == "asked" => total.pipelines += 1,
                "drawn" => {
                    total.first_drawn =
                        Some(total.first_drawn.map_or(frame, |first| first.min(frame)));
                }
                _ => {}
            }
        }
    }
    println!(
        "{}",
        json!({ "run": run.name, "causes": traced, "dropped_events": dropped })
    );
    for (cause, total) in totals {
        let (origin, what, frame) = match (cause, total.opened) {
            (0, _) => (
                json!("steady"),
                json!("work with no origin"),
                json!({ "unavailable": "steady_is_never_opened" }),
            ),
            (_, Some((origin, what, frame))) => (json!(origin), json!(what), json!(frame)),
            (_, None) => {
                let unknown = json!({ "unavailable": "opened_before_the_record" });
                (unknown.clone(), unknown.clone(), unknown)
            }
        };
        let bytes = if cause == 0 {
            json!({ "unavailable": "writes_with_no_cause_are_not_events" })
        } else {
            json!(total.bytes)
        };
        let first_drawn = total.first_drawn.map_or_else(
            || json!({ "unavailable": "no_first_draw_in_the_record" }),
            |frame| json!(frame),
        );
        println!(
            "{}",
            json!({
                "cause": cause,
                "origin": origin,
                "what": what,
                "frame": frame,
                "jobs": total.jobs,
                "ran_s": total.ran_s,
                "bytes_written": bytes,
                "pipelines": total.pipelines,
                "first_drawn_frame": first_drawn,
            })
        );
    }
    Ok(())
}

pub fn folder(root: &Path, name: &str) -> Result<PathBuf, Failure> {
    if name.is_empty() || name == "." || name == ".." || name.contains(['/', '\\', ':']) {
        return Err(json!({
            "code": "invalid_arguments",
            "path": "run",
            "got": name,
            "expected": "a folder name without separators",
        })
        .to_string()
        .into());
    }
    Ok(root.join("artifacts").join("runs").join(name))
}

pub fn read(root: &Path, name: &str) -> Result<Run, Failure> {
    let folder = folder(root, name)?;
    let path = folder.join("record.jsonl");
    let text = fs::read_to_string(&path).map_err(|error| {
        Failure::found(format!(
            "run {name}: cannot read {}: {error}",
            path.display()
        ))
    })?;
    let mut lines = text
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(at, line)| {
            serde_json::from_str::<Value>(line).map_err(|error| {
                Failure::from(format!(
                    "run {name}: line {} of record.jsonl: {error}",
                    at + 1
                ))
            })
        });
    let header = lines
        .next()
        .ok_or_else(|| Failure::from(format!("run {name}: record.jsonl is empty")))??;
    let entries = lines.collect::<Result<Vec<_>, _>>()?;
    Ok(Run {
        name: name.to_owned(),
        folder,
        header,
        entries,
    })
}

fn list(root: &Path) -> Result<(), Failure> {
    let base = root.join("artifacts").join("runs");
    let Ok(folders) = fs::read_dir(&base) else {
        return Ok(());
    };
    let mut names: Vec<String> = folders
        .flatten()
        .filter(|item| item.path().is_dir())
        .map(|item| item.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    for name in names {
        let record = match read(root, &name) {
            Ok(run) => {
                let count = |key: &str| {
                    run.entries
                        .iter()
                        .filter(|entry| !entry[key].is_null())
                        .count()
                };
                json!({
                    "entries": run.entries.len(),
                    "marks": count("mark"),
                    "pictures": count("picture"),
                    "host": run.header["host"],
                    "scene": run.header["scene"],
                    "pack": run.header["pack"],
                    "build": run.header["build"],
                })
            }
            Err(_) if !base.join(&name).join("record.jsonl").is_file() => {
                json!({ "unavailable": "no_record" })
            }
            Err(_) => json!({ "unavailable": "unreadable_record" }),
        };
        let open = base.join(&name).join("session.json").is_file();
        println!(
            "{}",
            json!({ "name": name, "open": open, "record": record })
        );
    }
    Ok(())
}

fn delete(root: &Path, name: &str) -> Result<(), Failure> {
    let folder = folder(root, name)?;
    if !folder.is_dir() {
        return Err(Failure::found(format!(
            "no run named {name} in {}",
            folder.display()
        )));
    }
    fs::remove_dir_all(&folder)
        .map_err(|error| format!("cannot delete {}: {error}", folder.display()))?;
    Ok(())
}
