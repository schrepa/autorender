use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Map, Value, json};

use super::matrix::Cell;
use super::session::{Run, SETTINGS, Shot};
use super::sides::{Side, checkout_tree};
use crate::bench::Scene;
use crate::heavy::HELD;
use crate::measure::{self, git, read_json};
use crate::process::Project;
use crate::serve::Server;

const PUBLIC: &str = "web/public";

pub(super) struct Page {
    pub(super) server: Server,
    pub(super) withheld: Vec<String>,
}

pub(super) fn serve(root: &Path, side: &Side, name: &str, folder: &Path) -> Result<Page, String> {
    let tree = match &side.sha {
        Some(sha) => checkout_tree(root, name, sha)?,
        None => root.to_path_buf(),
    };
    let ignored = git(
        &tree,
        &[
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--directory",
            "--",
            PUBLIC,
        ],
    )?;
    let withheld = ignored
        .lines()
        .filter_map(|path| path.strip_prefix(PUBLIC))
        .map(str::to_owned)
        .collect();
    eprintln!(
        "{}",
        json!({"event": "serving_browser_leg", "side": side.label})
    );
    let server = Server::page(&tree, &folder.join(format!("server-{name}.log")))?;
    Ok(Page { server, withheld })
}

pub(super) fn size(root: &Path) -> Result<[u32; 2], String> {
    let settings = read_json(&root.join(SETTINGS))?;
    settings["options"]["size"]
        .as_str()
        .and_then(|size| size.split_once('x'))
        .and_then(|(width, height)| Some([width.parse().ok()?, height.parse().ok()?]))
        .ok_or_else(|| format!("{SETTINGS} names no size WIDTHxHEIGHT"))
}

fn page_parameters(cell: &Cell, side: &Side) -> Result<Map<String, Value>, String> {
    let mut parameters = Map::new();
    parameters.insert("workers".into(), json!("off"));
    let settings = read_json(&side.tree.join(SETTINGS))?;
    let scene = Scene::read(&side.tree, &cell.scene)?;
    for (name, value) in settings["options"]
        .as_object()
        .into_iter()
        .flatten()
        .chain(&scene.options)
    {
        if name != "size" {
            parameters.insert(name.clone(), switch(value));
        }
    }
    for (option, value) in &cell.options {
        parameters.insert(option.clone(), json!(value));
    }
    Ok(parameters)
}

fn game_page(root: &Path, cell: &Cell) -> Result<String, String> {
    let project = Project::read(root, &cell.project)?;
    project
        .client()
        .and_then(|client| client.page.clone())
        .ok_or_else(|| format!("{}: project {} serves no page", cell.name, cell.project))
}

fn switch(value: &Value) -> Value {
    match value {
        Value::Bool(true) => json!("on"),
        Value::Bool(false) => json!("off"),
        other => other.clone(),
    }
}

pub(super) fn round(
    root: &Path,
    cell: &Cell,
    runs: &[(&Side, &Page, &str, usize)],
    size: [u32; 2],
    folder: &Path,
) -> Vec<Run> {
    let planned = || -> Result<Vec<Value>, String> {
        let html = game_page(root, cell)?;
        runs.iter()
            .map(|(side, page, role, index)| {
                Ok(json!({
                    "cell": cell.name,
                    "side": role,
                    "index": index,
                    "url": page.server.url,
                    "html": html,
                    "withheld": page.withheld,
                    "page": page_parameters(cell, side)?,
                    "environment": cell.environment,
                    "ticks": cell.ticks,
                    "views": cell.views.iter().map(|view| json!({"name": view.name, "camera": view.camera})).collect::<Vec<_>>(),
                    "captures": cell.captures.iter().map(|capture| capture.label()).collect::<Vec<_>>(),
                }))
            })
            .collect()
    };
    let failed = |error: String| {
        runs.iter()
            .map(|_| Run {
                error: Some(error.clone()),
                ..Run::default()
            })
            .collect()
    };
    let entries = match planned() {
        Ok(entries) => entries,
        Err(error) => return failed(error),
    };
    if let Err(error) = fs::create_dir_all(folder) {
        return failed(error.to_string());
    }
    let index = runs.first().map_or(0, |(_, _, _, index)| *index);
    let plan = folder.join(format!("browser-plan-{index}.json"));
    let result = folder.join(format!("browser-result-{index}.json"));
    let written = fs::write(
        &plan,
        json!({"size": size, "output": folder.parent().unwrap_or(folder), "runs": entries})
            .to_string(),
    );
    if let Err(error) = written {
        return failed(error.to_string());
    }
    let status = Command::new("node")
        .arg("scripts/equivalence-browser.mjs")
        .arg(&plan)
        .arg(&result)
        .env(HELD.0, HELD.1)
        .current_dir(root.join("web"))
        .status();
    match status {
        Ok(status) if status.success() => {}
        Ok(status) => return failed(format!("the browser driver exited {status}")),
        Err(error) => return failed(format!("cannot run node: {error}")),
    }
    match read_json(&result) {
        Ok(result) => result["runs"]
            .as_array()
            .into_iter()
            .flatten()
            .map(run_record)
            .collect(),
        Err(error) => failed(error),
    }
}

fn run_record(reported: &Value) -> Run {
    let mut run = Run {
        error: reported["error"].as_str().map(str::to_owned),
        ..Run::default()
    };
    for (key, shot) in reported["shots"].as_object().into_iter().flatten() {
        let shot = match (shot["unavailable"].as_str(), shot["path"].as_str()) {
            (Some(why), _) => Shot::Unavailable(why.to_owned()),
            (None, Some(path)) => match measure::file_hash(Path::new(path)) {
                Ok(sha256) => Shot::Taken {
                    path: PathBuf::from(path),
                    sha256,
                    work: measure::work_counters(&shot["rendering"]),
                },
                Err(error) => Shot::Unavailable(error),
            },
            (None, None) => Shot::Unavailable("the driver kept no file".into()),
        };
        run.shots.insert(key.clone(), shot);
    }
    for (view, digest) in reported["digests"].as_object().into_iter().flatten() {
        run.digests.insert(view.clone(), digest.clone());
    }
    run.unsettled = reported["unsettled"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    run
}
