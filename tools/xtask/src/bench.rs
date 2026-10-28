use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::Value;
use sha2::{Digest, Sha256};

pub const MEASURED: &[&str] = &[
    "frame_ms_mean",
    "frame_ms_p95",
    "frame_ms_pipelined_mean",
    "wall:frame_ms_mean",
    "load:prepare_ms",
    "gpu:*",
    "cpu:*",
    "mem:main_stack_peak_bytes",
    "mem:live_requested_bytes",
    "mem:peak_requested_bytes",
];

pub const NOT_WORK: &[&str] = &[
    "frame",
    "tick",
    "read_back_age",
    "simulation_ms",
    "extraction_ms",
    "render_ms",
    "resolution.gpu_ms",
];

const SCENE_OPTIONS: &[&str] = &["preset", "config", "view"];

const SETTINGS_OPTIONS: &[&str] = &["size", "jobcache", "rocks"];

const GENERATED: &[(&str, &[&str])] = &[(
    "stress/dense",
    &[
        "-p",
        "engine_vgbuild",
        "--profile",
        "quick",
        "--locked",
        "--bin",
        "dense_assets",
        "--",
    ],
)];

pub fn bench(root: &Path, options: &[String]) -> Result<(), String> {
    match options.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        ["check"] => check(root),
        ["assets"] => {
            let assets = assets(root)?;
            for (set, replaced) in install(root, &assets.folder)? {
                println!("{set}: {replaced} file(s) written to web/public/{set}");
            }
            println!(
                "{}",
                serde_json::json!({
                    "folder": assets.folder.display().to_string(),
                    "generated_hash": assets.generated_hash,
                })
            );
            Ok(())
        }
        _ => Err("usage: cargo xtask bench check | assets | budgets ...".into()),
    }
}

pub struct Assets {
    pub folder: PathBuf,
    pub generated_hash: String,
}

pub fn assets(root: &Path) -> Result<Assets, String> {
    let folder = build_sets(root, "bench-assets", GENERATED)?;
    let mut files = Vec::new();
    for (set, _) in GENERATED {
        files.extend(files_under(&folder, &folder.join(set))?);
    }
    files.sort();
    let mut hash = Sha256::new();
    for path in &files {
        let bytes =
            fs::read(folder.join(path)).map_err(|error| format!("cannot read {path}: {error}"))?;
        hash.update(path.as_bytes());
        hash.update([0]);
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(&bytes);
    }
    Ok(Assets {
        folder,
        generated_hash: hex(&hash.finalize()),
    })
}

pub fn dense_assets(root: &Path) -> Result<PathBuf, String> {
    build_sets(root, "dense-assets", &GENERATED[..1])
}

fn build_sets(
    root: &Path,
    cache: &str,
    sets: &[(&'static str, &'static [&'static str])],
) -> Result<PathBuf, String> {
    let inputs = inputs_hash(root, sets)?;
    let folder = root.join("target").join(cache).join(&inputs[..16]);
    let complete = folder.join("complete");
    if !complete.is_file() {
        if folder.exists() {
            fs::remove_dir_all(&folder)
                .map_err(|error| format!("cannot clear {}: {error}", folder.display()))?;
        }
        for (set, command) in sets {
            let into = folder.join(set);
            fs::create_dir_all(&into)
                .map_err(|error| format!("cannot create {}: {error}", into.display()))?;
            let into = into.display().to_string();
            let mut arguments = vec!["run"];
            arguments.extend(*command);
            arguments.push(&into);
            crate::process::run(root, "cargo", &arguments)?;
        }
        fs::write(&complete, &inputs)
            .map_err(|error| format!("cannot write {}: {error}", complete.display()))?;
        prune_assets(&folder)?;
    }
    Ok(folder)
}

const KEPT_ASSETS: usize = 3;

fn prune_assets(newest: &Path) -> Result<(), String> {
    let Some(parent) = newest.parent() else {
        return Ok(());
    };
    let mut others: Vec<(std::time::SystemTime, PathBuf)> = fs::read_dir(parent)
        .map_err(|error| format!("cannot list {}: {error}", parent.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path != newest)
        .map(|path| {
            let built = fs::metadata(path.join("complete"))
                .and_then(|metadata| metadata.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            (built, path)
        })
        .collect();
    others.sort_by_key(|(built, _)| std::cmp::Reverse(*built));
    for (_, path) in others.into_iter().skip(KEPT_ASSETS) {
        fs::remove_dir_all(&path)
            .map_err(|error| format!("cannot remove {}: {error}", path.display()))?;
    }
    Ok(())
}

fn inputs_hash(root: &Path, sets: &[(&str, &[&str])]) -> Result<String, String> {
    let generators: Vec<&str> = sets
        .iter()
        .filter_map(|(_, command)| {
            command
                .windows(2)
                .find(|pair| pair[0] == "-p")
                .map(|pair| pair[1])
        })
        .collect();
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--locked"])
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot run cargo metadata: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let metadata: Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("cargo metadata is not JSON: {error}"))?;
    let packages: BTreeMap<&str, &Value> = metadata["packages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|package| Some((package["id"].as_str()?, package)))
        .collect();
    let nodes: BTreeMap<&str, &Value> = metadata["resolve"]["nodes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|node| Some((node["id"].as_str()?, node)))
        .collect();
    let mut queue: VecDeque<&str> = packages
        .iter()
        .filter(|(_, package)| generators.contains(&package["name"].as_str().unwrap_or("")))
        .map(|(id, _)| *id)
        .collect();
    let mut seen = BTreeSet::new();
    let mut folders = BTreeSet::new();
    while let Some(id) = queue.pop_front() {
        if !seen.insert(id) {
            continue;
        }
        let package = packages[id];
        if package["source"].is_null() {
            let manifest = Path::new(package["manifest_path"].as_str().unwrap_or_default());
            if let Some(folder) = manifest.parent() {
                folders.insert(folder.to_path_buf());
            }
        }
        for dependency in nodes
            .get(id)
            .and_then(|node| node["deps"].as_array())
            .into_iter()
            .flatten()
        {
            let normal = dependency["dep_kinds"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|kind| kind["kind"].is_null() || kind["kind"] == "build");
            if let (true, Some(package)) = (normal, dependency["pkg"].as_str()) {
                queue.push_back(package);
            }
        }
    }
    let mut paths: Vec<String> = Vec::new();
    for folder in &folders {
        let relative = folder.strip_prefix(root).unwrap_or(folder);
        let listed = Command::new("git")
            .args(["ls-files", "-z", "--"])
            .arg(relative)
            .current_dir(root)
            .output()
            .map_err(|error| format!("cannot run git ls-files: {error}"))?;
        paths.extend(
            String::from_utf8_lossy(&listed.stdout)
                .split('\0')
                .filter(|path| !path.is_empty())
                .map(str::to_owned),
        );
    }
    paths.push("Cargo.lock".into());
    paths.sort();
    paths.dedup();
    let mut hash = Sha256::new();
    for path in &paths {
        let bytes = fs::read(root.join(path)).unwrap_or_default();
        hash.update(path.as_bytes());
        hash.update([0]);
        hash.update(normalized(&bytes));
        hash.update([0]);
    }
    let rustc = Command::new("rustc")
        .arg("-V")
        .output()
        .map_err(|error| format!("cannot run rustc: {error}"))?;
    hash.update(&rustc.stdout);
    for (set, command) in sets {
        hash.update(set.as_bytes());
        hash.update(command.join(" ").as_bytes());
    }
    Ok(hex(&hash.finalize()))
}

fn normalized(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    for (index, &byte) in bytes.iter().enumerate() {
        if !(byte == b'\r' && bytes.get(index + 1) == Some(&b'\n')) {
            out.push(byte);
        }
    }
    out
}

fn files_under(base: &Path, folder: &Path) -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    let entries = fs::read_dir(folder)
        .map_err(|error| format!("cannot read {}: {error}", folder.display()))?;
    for entry in entries {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_dir() {
            files.extend(files_under(base, &path)?);
        } else {
            let relative = path.strip_prefix(base).unwrap_or(&path);
            files.push(
                relative
                    .components()
                    .map(|part| part.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/"),
            );
        }
    }
    Ok(files)
}

pub fn install(root: &Path, folder: &Path) -> Result<Vec<(&'static str, usize)>, String> {
    let mut installed = Vec::new();
    for (set, _) in GENERATED {
        installed.push((
            *set,
            copy_changed(&folder.join(set), &root.join("web/public").join(set))?,
        ));
    }
    Ok(installed)
}

fn copy_changed(from: &Path, into: &Path) -> Result<usize, String> {
    fs::create_dir_all(into)
        .map_err(|error| format!("cannot create {}: {error}", into.display()))?;
    let mut copied = 0;
    for path in files_under(from, from)? {
        let (source, target) = (from.join(&path), into.join(&path));
        let bytes = fs::read(&source).map_err(|error| format!("cannot read {path}: {error}"))?;
        if fs::read(&target).is_ok_and(|existing| existing == bytes) {
            continue;
        }
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        fs::write(&target, bytes)
            .map_err(|error| format!("cannot write {}: {error}", target.display()))?;
        copied += 1;
    }
    Ok(copied)
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn read(root: &Path, path: &str) -> Result<Value, String> {
    let text = fs::read_to_string(root.join("bench").join(path))
        .map_err(|error| format!("cannot read bench/{path}: {error}"))?;
    serde_json::from_str(&text).map_err(|error| format!("bench/{path}: {error}"))
}

pub fn catalog(root: &Path) -> Result<Vec<Value>, String> {
    read(root, "metrics.json")?["metrics"]
        .as_array()
        .cloned()
        .ok_or_else(|| "bench/metrics.json: `metrics` is a list".into())
}

pub fn covers(pattern: &str, name: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => name.starts_with(prefix),
        None => pattern == name,
    }
}

fn stats_leaves() -> Vec<String> {
    let schema: Value =
        serde_json::from_str(engine_dev::FRAME_STATS_SCHEMA).expect("generated schema is JSON");
    let definitions = schema["$defs"].clone();
    let mut leaves = Vec::new();
    fn walk(node: &Value, definitions: &Value, path: &str, leaves: &mut Vec<String>) {
        let node = resolve(node, definitions);
        if let Some(properties) = node["properties"].as_object() {
            for (name, property) in properties {
                let path = if path.is_empty() {
                    name.clone()
                } else {
                    format!("{path}.{name}")
                };
                walk(property, definitions, &path, leaves);
            }
        } else {
            leaves.push(path.to_owned());
        }
    }
    fn resolve<'a>(node: &'a Value, definitions: &'a Value) -> &'a Value {
        if let Some(reference) = node["$ref"].as_str() {
            let name = reference.rsplit('/').next().unwrap_or_default();
            return resolve(&definitions[name], definitions);
        }
        for key in ["anyOf", "oneOf", "allOf"] {
            if let Some(options) = node[key].as_array() {
                let options: Vec<&Value> = options
                    .iter()
                    .filter(|option| option["type"] != "null")
                    .collect();
                if let [only] = options[..] {
                    return resolve(only, definitions);
                }
            }
        }
        node
    }
    walk(&schema, &definitions, "", &mut leaves);
    leaves
}

pub fn check(root: &Path) -> Result<(), String> {
    let mut problems = Vec::new();
    let catalog = catalog(root)?;
    let mut names = BTreeSet::new();
    for (index, entry) in catalog.iter().enumerate() {
        let name = entry["name"].as_str().unwrap_or_default();
        let at = format!("bench/metrics.json metrics[{index}] {name}");
        if name.is_empty() || !names.insert(name.to_owned()) {
            problems.push(format!("{at}: a name, once"));
        }
        let kind = entry["kind"].as_str().unwrap_or_default();
        if !["timing", "work", "counter", "memory"].contains(&kind) {
            problems.push(format!("{at}: kind is timing, work, counter or memory"));
        }
        let rule = entry["rule"].as_str();
        if (kind == "work")
            != rule.is_some_and(|rule| ["equal", "not-above", "false", "shown"].contains(&rule))
        {
            problems.push(format!(
                "{at}: work counters, and only they, have a rule: equal, not-above, false or shown"
            ));
        }
        for field in ["statistic", "unit"] {
            if entry[field].as_str().is_none_or(str::is_empty) {
                problems.push(format!("{at}: {field} says what it is"));
            }
        }
        let runtimes = entry["runtimes"].as_array();
        if runtimes.is_none_or(|runtimes| {
            runtimes.is_empty()
                || runtimes
                    .iter()
                    .any(|runtime| runtime != "native" && runtime != "browser")
        }) {
            problems.push(format!("{at}: runtimes lists native and/or browser"));
        }
        if !matches!(&entry["direction"], Value::Null)
            && !["lower", "higher"].contains(&entry["direction"].as_str().unwrap_or_default())
        {
            problems.push(format!("{at}: direction is lower, higher or null"));
        }
    }
    let measured: BTreeSet<&str> = catalog
        .iter()
        .filter(|entry| entry["kind"] != "work")
        .filter_map(|entry| entry["name"].as_str())
        .collect();
    let reported: BTreeSet<&str> = MEASURED.iter().copied().collect();
    for name in measured.difference(&reported) {
        problems.push(format!(
            "bench/metrics.json lists {name}, which no measurement reports"
        ));
    }
    for name in reported.difference(&measured) {
        problems.push(format!(
            "measurements report {name}, which bench/metrics.json does not list"
        ));
    }
    let work: BTreeSet<String> = catalog
        .iter()
        .filter(|entry| entry["kind"] == "work")
        .filter_map(|entry| entry["name"].as_str().map(str::to_owned))
        .collect();
    let leaves: BTreeSet<String> = stats_leaves()
        .into_iter()
        .filter(|leaf| !NOT_WORK.contains(&leaf.as_str()))
        .collect();
    for name in work.difference(&leaves) {
        problems.push(format!(
            "bench/metrics.json lists work counter {name}, which the frame statistics lack"
        ));
    }
    for name in leaves.difference(&work) {
        problems.push(format!(
            "the frame statistics' {name} is not in bench/metrics.json (a work counter with its rule)"
        ));
    }
    problems.extend(option_file_problems(root)?);
    problems.extend(shader_path_problems(root)?);
    let covered = |metric: &str| {
        catalog
            .iter()
            .any(|entry| covers(entry["name"].as_str().unwrap_or_default(), metric))
    };
    problems.extend(crate::budgets::problems(root, &covered)?);
    if problems.is_empty() {
        println!(
            "bench/ matches the code: {} catalog entries, the option files, shader paths and budgets.",
            catalog.len()
        );
        Ok(())
    } else {
        Err(format!(
            "bench/ disagrees with the code:\n  {}",
            problems.join("\n  ")
        ))
    }
}

fn option_file_problems(root: &Path) -> Result<Vec<String>, String> {
    let switches = read(root, "options.json")?["options"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let presets = engine_presets(root)?;
    let mut problems = Vec::new();
    for (folder, allowed) in [("scenes", SCENE_OPTIONS), ("settings", SETTINGS_OPTIONS)] {
        let entries = fs::read_dir(root.join("bench").join(folder))
            .map_err(|error| format!("cannot read bench/{folder}: {error}"))?;
        for entry in entries {
            let name = entry
                .map_err(|error| error.to_string())?
                .file_name()
                .to_string_lossy()
                .into_owned();
            let path = format!("{folder}/{name}");
            let file = read(root, &path)?;
            let at = format!("bench/{path}");
            if !name.ends_with(".json") {
                problems.push(format!("{at}: option files end in .json"));
            }
            if file["note"].as_str().is_none_or(str::is_empty) {
                problems.push(format!("{at}: a note says what it is"));
            }
            let Some(options) = file["options"].as_object() else {
                problems.push(format!("{at}: options is an object of options by name"));
                continue;
            };
            if let Some(field) = file
                .as_object()
                .into_iter()
                .flat_map(|fields| fields.keys())
                .find(|field| !["note", "options"].contains(&field.as_str()))
            {
                problems.push(format!("{at}: {field} is not a field (note, options)"));
            }
            for (option, value) in options {
                let switch = switches.iter().find(|switch| switch["name"] == *option);
                match switch {
                    Some(switch) if folder == "settings" => {
                        let values = switch["values"].as_array().cloned().unwrap_or_default();
                        if !values.contains(value) {
                            problems.push(format!(
                                "{at}: {option} is one of bench/options.json's values, not {value}"
                            ));
                        }
                    }
                    _ if allowed.contains(&option.as_str()) => {}
                    _ => problems.push(format!(
                        "{at}: {option} does not belong here ({})",
                        allowed.join(", ")
                    )),
                }
            }
            if folder == "scenes" {
                let preset = options["preset"].as_str().unwrap_or_default();
                if !presets.iter().any(|known| known == preset) {
                    problems.push(format!(
                        "{at}: preset {preset} is not one of {}",
                        presets.join(", ")
                    ));
                }
                if !options["config"].is_object() {
                    problems.push(format!("{at}: config is the whole workload configuration"));
                }
            }
        }
    }
    Ok(problems)
}

fn engine_presets(root: &Path) -> Result<Vec<String>, String> {
    stress_choices(root, "presets")
}

pub fn stress_choices(root: &Path, list: &str) -> Result<Vec<String>, String> {
    let page = fs::read_to_string(root.join("web/src/generated/stress.ts"))
        .map_err(|error| format!("cannot read web/src/generated/stress.ts: {error}"))?;
    let start = page
        .find("export const STRESS_CATALOG = ")
        .map(|at| at + "export const STRESS_CATALOG = ".len())
        .ok_or("web/src/generated/stress.ts has no STRESS_CATALOG")?;
    let end = page[start..]
        .find(" as const;")
        .ok_or("STRESS_CATALOG does not end")?;
    let catalog: Value = serde_json::from_str(&page[start..start + end])
        .map_err(|error| format!("STRESS_CATALOG: {error}"))?;
    Ok(catalog[list]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|choice| choice["value"].as_str().map(str::to_owned))
        .collect())
}

fn shader_path_problems(root: &Path) -> Result<Vec<String>, String> {
    let patterns: Vec<String> = serde_json::from_value(read(root, "shader-paths.json")?)
        .map_err(|error| format!("bench/shader-paths.json is a list of path patterns: {error}"))?;
    let listed = Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot run git ls-files: {error}"))?;
    let files: Vec<String> = String::from_utf8_lossy(&listed.stdout)
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(str::to_owned)
        .collect();
    let mut problems = Vec::new();
    for file in files.iter().filter(|file| file.ends_with(".wgsl")) {
        if !patterns.iter().any(|pattern| glob(pattern, file)) {
            problems.push(format!(
                "{file} is a shader no pattern in bench/shader-paths.json covers"
            ));
        }
    }
    for pattern in &patterns {
        if !files.iter().any(|file| glob(pattern, file)) {
            problems.push(format!(
                "bench/shader-paths.json's {pattern} covers no file"
            ));
        }
    }
    Ok(problems)
}

pub fn glob(pattern: &str, path: &str) -> bool {
    fn segments(pattern: &[&str], path: &[&str]) -> bool {
        match (pattern.first(), path.first()) {
            (None, None) => true,
            (Some(&"**"), _) => {
                segments(&pattern[1..], path) || (!path.is_empty() && segments(pattern, &path[1..]))
            }
            (Some(part), Some(name)) => segment(part, name) && segments(&pattern[1..], &path[1..]),
            _ => false,
        }
    }
    fn segment(pattern: &str, name: &str) -> bool {
        match pattern.split_once('*') {
            None => pattern == name,
            Some((head, rest)) => {
                name.starts_with(head)
                    && (head.len()..=name.len())
                        .any(|at| name.is_char_boundary(at) && segment(rest, &name[at..]))
            }
        }
    }
    let pattern: Vec<&str> = pattern.split('/').collect();
    let path: Vec<&str> = path.split('/').collect();
    segments(&pattern, &path)
}

pub(crate) fn generated(root: &Path, source: &Path, sha: &str) -> Result<(String, String), String> {
    eprintln!("Generating the files workloads read at {sha}.");
    let target = root.join("target").join("measure").join("target");
    let made = Command::new("cargo")
        .args(["xtask", "bench", "assets"])
        .env("CARGO_TARGET_DIR", &target)
        .current_dir(source)
        .stderr(Stdio::inherit())
        .output()
        .map_err(|error| format!("cannot run cargo: {error}"))?;
    let printed = String::from_utf8_lossy(&made.stdout);
    let summary = printed
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|_| made.status.success());
    match summary {
        Some(summary) => match (
            summary["folder"].as_str(),
            summary["generated_hash"].as_str(),
        ) {
            (Some(folder), Some(hash)) => Ok((folder.to_owned(), hash.to_owned())),
            _ => Err(format!(
                "`cargo xtask bench assets` at {sha} printed no folder and generated_hash"
            )),
        },
        None => Err(format!(
            "`cargo xtask bench assets` at {sha} failed ({}): measured commits need it",
            made.status
        )),
    }
}
