use std::{
    fs,
    path::{Path, PathBuf},
};

use serde_json::{Value, json};

use crate::Failure;

pub struct Run {
    pub name: String,
    pub folder: PathBuf,
    pub header: Value,
    pub entries: Vec<Value>,
}

pub fn runs(root: &Path, arguments: &[String]) -> Result<(), Failure> {
    match arguments {
        [] => list(root),
        [verb, name] if verb == "delete" => delete(root, name),
        _ => Err(json!({
            "code": "invalid_arguments",
            "path": "runs",
            "got": arguments.join(" "),
            "expected": "runs | runs delete <name>",
        })
        .to_string()
        .into()),
    }
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
            Err(failure) => json!({ "unavailable": failure.message }),
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
