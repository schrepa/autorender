use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

use crate::Failure;
use crate::heavy::HeavyRun;
use crate::pipeline::QUICK;
use crate::process::{Project, engine_commit, project_option, run};

const COOKED: &str = "web/public/cooked";

const READ_BACK: &str = "target/cook";

const PLAYER: &str = "crates/engine_player";

const CHECK_FRAMES: &str = "600";

const USAGE: &str = "usage: cargo xtask cook [--project DIR] [PATH.scene.json...] [--check]";

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
            arguments.extend(["--source", scene.as_str()]);
        }
        arguments.extend(rest);
        arguments
    }
}

pub fn cook(root: &Path, options: &[String]) -> Result<(), Failure> {
    let (project, rest) = project_option(options)?;
    let mut check = false;
    let mut scenes = Vec::new();
    for option in rest {
        match option.as_str() {
            "--check" => check = true,
            flag if flag.starts_with("--") => {
                return Err(format!("unknown option {flag}\n{USAGE}").into());
            }
            _ => scenes.push(option.replace('\\', "/")),
        }
    }
    let mut packs = Vec::new();
    if let Some(folder) = &project {
        let project = Project::read(root, folder)?;
        packs.push(Pack {
            name: project.package.clone(),
            program: build(root, &project)?,
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
        let program = build(root, &Project::read(root, PLAYER)?)?;
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
    let names: Vec<&str> = packs.iter().map(|pack| pack.name.as_str()).collect();
    let _lock = HeavyRun::hold(&format!("cook {}", names.join(" ")))?;
    if check {
        check_packs(root, &packs)?;
    } else {
        for pack in &packs {
            let file = pack.file();
            let arguments = pack.arguments(&["--packs", COOKED, "--cook", &file]);
            run(root, &pack.program, &arguments)?;
        }
    }
    Ok(write_index(&folder)?)
}

fn build(root: &Path, project: &Project) -> Result<String, String> {
    run(
        root,
        "cargo",
        &[
            "build",
            "-p",
            &project.package,
            "--bin",
            &project.native,
            "--profile",
            QUICK,
            "--locked",
        ],
    )?;
    let program = root.join("target").join(QUICK).join(project.program());
    Ok(program.to_string_lossy().into_owned())
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

fn check_packs(root: &Path, packs: &[Pack]) -> Result<(), Failure> {
    crate::pipeline::build_wasm(root, true, QUICK)?;
    let mut problems = Vec::new();
    for pack in packs {
        let read = read_back(root, pack, &mut problems)?;
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
        run_on_pack(root, pack, &mut problems)?;
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
                "hash": crate::pure::hash(&record),
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

fn run_on_pack(root: &Path, pack: &Pack, problems: &mut Vec<Value>) -> Result<(), String> {
    let file = pack.file();
    let arguments = pack.arguments(&[
        "--packs",
        COOKED,
        "--offscreen",
        CHECK_FRAMES,
        "--size",
        "640x360",
        "--jobcache",
        "verify",
    ]);
    let record = json!({
        "event": "running",
        "program": pack.program,
        "arguments": arguments,
    });
    println!("{record}");
    let output = Command::new(&pack.program)
        .args(&arguments)
        .envs([engine_commit()])
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
