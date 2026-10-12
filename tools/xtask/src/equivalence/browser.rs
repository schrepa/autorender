//! The browser leg (`--browser`, `--both`): the same cells run in headless
//! Chrome against a development server of each side, driven by
//! `web/scripts/equivalence-browser.mjs`, and judged as the native leg's
//! runs are.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Map, Value, json};

use super::matrix::{Cell, Workload};
use super::session::{Run, SETTINGS, Shot, scene_options};
use super::sides::{Side, checkout_tree};
use crate::heavy::HELD;
use crate::measure::{self, read_json};
use crate::serve::Server;

/// A side's page, served from its own tree.
pub(super) struct Page {
    pub(super) server: Server,
}

/// Serves `side`'s page: the working tree in place, or its commit (`sha`)
/// checked out in the reused tree `name`, with the side's generated files
/// where the page reads them; logs into `folder`.
pub(super) fn serve(root: &Path, side: &Side, name: &str, folder: &Path) -> Result<Page, String> {
    let tree = match &side.sha {
        Some(sha) => checkout_tree(root, name, sha)?,
        None => root.to_path_buf(),
    };
    crate::bench::install(&tree, &side.assets)?;
    eprintln!("Serving {} for the browser leg.", side.label);
    let server = Server::page(&tree, &folder.join(format!("server-{name}.log")))?;
    Ok(Page { server })
}

/// The page size every cell draws at: the settings bundle's.
pub(super) fn size(root: &Path) -> Result<[u32; 2], String> {
    let settings = read_json(&root.join(SETTINGS))?;
    settings["options"]["size"]
        .as_str()
        .and_then(|size| size.split_once('x'))
        .and_then(|(width, height)| Some([width.parse().ok()?, height.parse().ok()?]))
        .ok_or_else(|| format!("{SETTINGS} names no size WIDTHxHEIGHT"))
}

/// The page parameters of `cell` on `side`: the settings bundle's switches,
/// the scene's options and the cell's, job workers off; a source cell's
/// scene comes from the driver.
fn page_parameters(cell: &Cell, side: &Side) -> Result<Map<String, Value>, String> {
    let mut parameters = Map::new();
    parameters.insert("workers".into(), json!("off"));
    let settings = read_json(&side.tree.join(SETTINGS))?;
    for (name, value) in settings["options"].as_object().into_iter().flatten() {
        // The page takes its size from the viewport; scanned rocks are left
        // out by the driver.
        if !["size", "rocks"].contains(&name.as_str()) {
            parameters.insert(name.clone(), switch(value));
        }
    }
    if let Workload::Scene(_) = cell.workload {
        parameters.extend(scene_options(cell, &side.tree)?);
    }
    for (option, value) in &cell.options {
        if option != "config" {
            parameters.insert(option.clone(), json!(value));
        }
    }
    Ok(parameters)
}

/// An option file's value as the page's parameter: switches as on and off.
fn switch(value: &Value) -> Value {
    match value {
        Value::Bool(true) => json!("on"),
        Value::Bool(false) => json!("off"),
        other => other.clone(),
    }
}

/// One round of `cell` in the browser: `runs` (each a side's page, its role
/// and the run's number), into `folder`; each run's record, in order.
pub(super) fn round(
    root: &Path,
    cell: &Cell,
    runs: &[(&Side, &Page, &str, usize)],
    size: [u32; 2],
    folder: &Path,
) -> Vec<Run> {
    let planned = || -> Result<Vec<Value>, String> {
        runs.iter()
            .map(|(side, page, role, index)| {
                Ok(json!({
                    "cell": cell.name,
                    "side": role,
                    "index": index,
                    "url": page.server.url,
                    "tree": side.tree.display().to_string(),
                    "page": page_parameters(cell, side)?,
                    "source": match &cell.workload {
                        Workload::Source(source) => json!(source),
                        Workload::Scene(_) => Value::Null,
                    },
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

/// A run the driver reported, as the judge reads it.
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
