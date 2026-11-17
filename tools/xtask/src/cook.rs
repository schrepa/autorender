use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::{Value, json};

use crate::Failure;
use crate::pipeline::build_native;
use crate::process::{Project, project_option, run};

const COOKED: &str = "web/public/cooked";

const READ_BACK: &str = "target/cook";

const PLAYER: &str = "crates/engine/examples/bare_game";

const CHECK_FRAMES: &str = "600";
const CHECK_SIZE: &str = "640x360";
const PASSES: [&str; 2] = ["records", "pipelines"];

const USAGE: &str = "usage: cargo xtask cook [--project DIR] [PATH.scene.json...] \
                     --limit-ms MS | --check [--sample N]";

struct Pack {
    name: String,
    program: String,
    scene: Option<String>,
}

impl Pack {
    fn file(&self) -> String {
        format!("{COOKED}/{}.pack", self.name)
    }

    fn arguments<'a>(&'a self, rest: &[&'a str]) -> Vec<&'a str> {
        let mut arguments = Vec::new();
        if let Some(scene) = &self.scene {
            arguments.extend(["--scene", scene.as_str()]);
        }
        arguments.extend(rest);
        arguments
    }
}

pub fn cook(root: &Path, options: &[String]) -> Result<(), Failure> {
    let (project, rest) = project_option(options)?;
    let mut check = false;
    let mut limit = None;
    let mut sample = None;
    let mut scenes = Vec::new();
    let mut given = rest.into_iter();
    while let Some(option) = given.next() {
        match option.as_str() {
            "--check" => check = true,
            "--limit-ms" => {
                limit = Some(
                    given
                        .next()
                        .filter(|limit| limit.parse::<u64>().is_ok_and(|limit| limit > 0))
                        .ok_or(USAGE)?,
                );
            }
            "--sample" => {
                sample = Some(
                    given
                        .next()
                        .filter(|count| count.parse::<u32>().is_ok_and(|count| count > 0))
                        .ok_or(USAGE)?,
                );
            }
            flag if flag.starts_with("--") => {
                return Err(format!("unknown option {flag}\n{USAGE}").into());
            }
            _ => scenes.push(option.replace('\\', "/")),
        }
    }
    if check == limit.is_some() || (sample.is_some() && !check) {
        return Err(USAGE.into());
    }
    let mut packs = Vec::new();
    if let Some(folder) = &project {
        let project = Project::read(root, folder)?;
        packs.push(Pack {
            name: project.package.clone(),
            program: build_native(root, &project)?,
            scene: None,
        });
        scenes.extend(
            project
                .scenes
                .iter()
                .map(|scene| format!("{folder}/{scene}")),
        );
    }
    if !scenes.is_empty() {
        let program = build_native(root, &Project::read(root, PLAYER)?)?;
        for scene in scenes {
            packs.push(Pack {
                name: scene_id(root, &scene)?,
                program: program.clone(),
                scene: Some(scene),
            });
        }
    }
    if packs.is_empty() {
        return Err(format!("name a project or scene files\n{USAGE}").into());
    }
    for (at, pack) in packs.iter().enumerate() {
        if let Some(other) = packs[..at].iter().find(|other| other.name == pack.name) {
            return Err(json!({
                "code": "duplicate_id",
                "path": "$.id",
                "message": format!("both cook into {}: give each its own id", pack.file()),
                "files": [cooked_by(other), cooked_by(pack)],
            })
            .to_string()
            .into());
        }
    }
    let folder = root.join(COOKED);
    fs::create_dir_all(&folder)
        .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
    match limit {
        None => check_packs(root, &packs, sample.as_deref())?,
        Some(limit) => {
            for pack in &packs {
                let file = pack.file();
                let arguments = pack.arguments(&[
                    "--packs",
                    COOKED,
                    "--cook",
                    &file,
                    "--offscreen",
                    CHECK_FRAMES,
                    "--size",
                    CHECK_SIZE,
                    "--limit-ms",
                    &limit,
                ]);
                for pass in PASSES {
                    println!(
                        "{}",
                        json!({ "event": "cook_pass", "pack": pack.name, "pass": pass })
                    );
                    run(root, &pack.program, &arguments)?;
                }
            }
        }
    }
    Ok(write_index(&folder)?)
}

pub fn project_packs(root: &Path, folder: &str, project: &Project) -> Result<Vec<PathBuf>, String> {
    let mut names = vec![project.package.clone()];
    for scene in &project.scenes {
        names.push(scene_id(root, &format!("{folder}/{scene}"))?);
    }
    let cooked = root.join(COOKED);
    Ok(names
        .iter()
        .map(|name| cooked.join(format!("{name}.pack")))
        .collect())
}

fn scene_id(root: &Path, scene: &str) -> Result<String, String> {
    let text = fs::read_to_string(root.join(scene))
        .map_err(|error| format!("cannot read the scene {scene}: {error}"))?;
    engine_scene::SceneDocument::from_json(&text)
        .map(|document| document.id)
        .map_err(|error| json!({ "file": scene, "error": error }).to_string())
}

fn cooked_by(pack: &Pack) -> String {
    pack.scene.clone().unwrap_or_else(|| pack.name.clone())
}

fn check_packs(root: &Path, packs: &[Pack], sample: Option<&str>) -> Result<(), Failure> {
    if sample.is_none() {
        crate::pipeline::build_wasm(root, true)?;
    }
    let mut problems = Vec::new();
    for pack in packs {
        let read = read_back(root, pack, &mut problems)?;
        if sample.is_none() {
            let listed = root
                .join(READ_BACK)
                .join(format!("{}.records.json", pack.name));
            if let Some(folder) = listed.parent() {
                fs::create_dir_all(folder)
                    .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
            }
            fs::write(&listed, Value::Array(read).to_string())
                .map_err(|error| format!("cannot write {}: {error}", listed.display()))?;
            read_in_wasm(root, pack, &listed, &mut problems)?;
        }
        run_on_pack(root, pack, sample, &mut problems)?;
    }
    if problems.is_empty() {
        return Ok(());
    }
    Err(Failure::found(format!(
        "{} problem(s) in the cooked packs:\n{}",
        problems.len(),
        problems
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    )))
}

fn read_back(root: &Path, pack: &Pack, problems: &mut Vec<Value>) -> Result<Vec<Value>, String> {
    let file = pack.file();
    let bytes =
        fs::read(root.join(&file)).map_err(|error| format!("cannot read {file}: {error}"))?;
    let header = bytes
        .get(..engine_worker::PACK_HEADER)
        .ok_or_else(|| format!("{file} has no header"))?;
    let range = |offset: u64, length: u64| -> Result<&[u8], String> {
        let start = usize::try_from(offset).map_err(|error| error.to_string())?;
        let end = start + usize::try_from(length).map_err(|error| error.to_string())?;
        bytes
            .get(start..end)
            .ok_or_else(|| format!("{file} ends before byte {end}"))
    };
    let (offset, stored, _) =
        engine_worker::pack_directory(header).map_err(|error| error.to_string())?;
    let index = engine_worker::PackIndex::open(header, range(offset, stored)?)
        .map_err(|error| error.to_string())?;
    let listing = engine_worker::listing_key();
    let entry = index
        .entries()
        .iter()
        .find(|entry| entry.key == listing)
        .ok_or_else(|| format!("{file} lists no record kinds: cook it again"))?;
    let kinds = engine_worker::read_listing(
        &engine_worker::unpack(entry, range(entry.offset, u64::from(entry.stored))?)
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let mut read = Vec::with_capacity(index.len());
    for entry in index.entries() {
        let key: String = entry.key.iter().map(|byte| format!("{byte:02x}")).collect();
        let kind = match kinds.get(&entry.key) {
            Some(kind) => kind.as_str(),
            None if entry.key == listing => engine_worker::LISTING_KIND,
            None => {
                problems.push(json!({
                    "pack": file,
                    "check": "listed",
                    "key": key,
                    "message": "the pack's listing names no kind for this record",
                }));
                continue;
            }
        };
        let unpacked = range(entry.offset, u64::from(entry.stored)).and_then(|stored| {
            engine_worker::unpack(entry, stored).map_err(|error| error.to_string())
        });
        match unpacked {
            Ok(record) => read.push(json!({
                "kind": kind,
                "key": key,
                "hash": crate::process::hash(&record),
            })),
            Err(error) => problems.push(json!({
                "pack": file,
                "check": "native",
                "kind": kind,
                "key": key,
                "message": error,
            })),
        }
    }
    let record = json!({
        "event": "pack_read_native",
        "pack": pack.name,
        "records": read.len(),
        "kinds": kinds
            .values()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
    });
    println!("{record}");
    Ok(read)
}

fn read_in_wasm(
    root: &Path,
    pack: &Pack,
    listed: &Path,
    problems: &mut Vec<Value>,
) -> Result<(), String> {
    let file = pack.file();
    let listed = listed.display().to_string();
    let record = json!({
        "event": "running",
        "program": "node",
        "arguments": ["web/scripts/pack-wasm.mjs", file, listed],
    });
    println!("{record}");
    let output = Command::new("node")
        .args(["web/scripts/pack-wasm.mjs", &file, &listed])
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot run node: {error}"))?;
    if !(output.status.success() || output.status.code() == Some(3)) {
        return Err(format!(
            "the Wasm read-back of {file} stopped ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let report: Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("the Wasm read-back of {file} printed no report: {error}"))?;
    let record = json!({
        "event": "pack_read_wasm",
        "pack": pack.name,
        "records": report["records"],
        "mismatched": report["mismatched"],
    });
    println!("{record}");
    let named = report["mismatches"].as_array().cloned().unwrap_or_default();
    for mismatch in &named {
        problems.push(json!({
            "pack": file,
            "check": "wasm",
            "kind": mismatch["kind"],
            "key": mismatch["key"],
            "native": mismatch["native"],
            "wasm": mismatch["wasm"],
        }));
    }
    let mismatched = report["mismatched"].as_u64().unwrap_or(u64::MAX);
    if mismatched > named.len() as u64 {
        problems.push(json!({
            "pack": file,
            "check": "wasm",
            "message": format!("{mismatched} records read back in Wasm differ from natively"),
        }));
    }
    Ok(())
}

fn run_on_pack(
    root: &Path,
    pack: &Pack,
    sample: Option<&str>,
    problems: &mut Vec<Value>,
) -> Result<(), String> {
    let file = pack.file();
    let cache = sample.map_or_else(|| "verify".to_owned(), |count| format!("verify:{count}"));
    let rest = [
        "--packs",
        COOKED,
        "--offscreen",
        CHECK_FRAMES,
        "--size",
        CHECK_SIZE,
        "--jobcache",
        &cache,
    ];
    let arguments = pack.arguments(&rest);
    let record = json!({
        "event": "running",
        "program": pack.program,
        "arguments": arguments,
    });
    println!("{record}");
    let output = Command::new(&pack.program)
        .args(&arguments)
        .current_dir(root)
        .stderr(Stdio::inherit())
        .output()
        .map_err(|error| format!("cannot run {}: {error}", pack.program))?;
    if !output.status.success() {
        return Err(format!(
            "{} did not run on its pack ({}); see its output above",
            pack.name, output.status
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let report: Value = text
        .find('{')
        .and_then(|start| serde_json::from_str(&text[start..]).ok())
        .ok_or_else(|| format!("{} printed no report", pack.name))?;
    let stats = &report["stats"];
    let jobs = &stats["jobs"];
    let (hits, mismatches) = (
        jobs["cache_hits"].as_u64().unwrap_or(0),
        jobs["cache_mismatches"].as_u64().unwrap_or(u64::MAX),
    );
    let (listed, unlisted) = (
        stats["pipelines_listed"].as_u64().unwrap_or(0),
        stats["pipelines_unlisted"].as_u64().unwrap_or(u64::MAX),
    );
    let record = json!({
        "event": "pack_run",
        "pack": pack.name,
        "records_read": hits,
        "mismatched": mismatches,
        "pipelines_listed": listed,
        "pipelines_unlisted": unlisted,
    });
    println!("{record}");
    let named = report["mismatched"].as_array().cloned().unwrap_or_default();
    for mismatch in &named {
        problems.push(json!({
            "pack": file,
            "check": "fresh",
            "kind": mismatch["kind"],
            "key": mismatch["key"],
            "message": "the record differs from a fresh run of what makes it",
        }));
    }
    if sample.is_some() {
        sampled(pack, &report, problems);
    }
    let mut fault = |check: &str, message: String| {
        problems.push(json!({ "pack": file, "check": check, "message": message }));
    };
    if hits == 0 {
        fault("read", "no record was read from the pack".into());
    }
    if mismatches > named.len() as u64 {
        fault(
            "fresh",
            format!("{mismatches} records differ from a fresh run of what makes them"),
        );
    }
    if listed == 0 {
        fault(
            "pipelines",
            "no pipeline was created from the pack's pipeline list".into(),
        );
    }
    if unlisted > 0 {
        fault(
            "pipelines",
            format!("{unlisted} pipelines were created that the pack's pipeline list lacks"),
        );
    }
    Ok(())
}

fn sampled(pack: &Pack, report: &Value, problems: &mut Vec<Value>) {
    let mut kinds: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for record in report["sampled"].as_array().into_iter().flatten() {
        *kinds
            .entry(record["kind"].as_str().unwrap_or("records"))
            .or_default() += 1;
    }
    let record = json!({ "event": "pack_sampled", "pack": pack.name, "kinds": kinds });
    println!("{record}");
    if kinds.is_empty() {
        problems.push(json!({
            "pack": pack.file(),
            "check": "sampled",
            "message": "the run read no sampled record, so none was made again",
        }));
    }
}

fn write_index(folder: &Path) -> Result<(), String> {
    let mut packs: Vec<String> = fs::read_dir(folder)
        .map_err(|error| format!("cannot list {}: {error}", folder.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".pack"))
        .collect();
    packs.sort();
    let index = folder.join("index.json");
    fs::write(
        &index,
        serde_json::to_string(&packs).expect("names_serialize") + "\n",
    )
    .map_err(|error| format!("cannot write {}: {error}", index.display()))
}
