use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::Repository;

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

const SCENE_FIELDS: &[&str] = &["note", "project", "options"];

const SETTINGS_FIELDS: &[&str] = &["note", "options"];

const HOST_OPTIONS: &[&str] = &["size", "jobcache"];

pub fn switch(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Bool(true) => "on".into(),
        Value::Bool(false) => "off".into(),
        other => other.to_string(),
    }
}

pub fn setting_values(option: &str) -> Option<&'static [&'static str]> {
    engine::host::SETTINGS
        .iter()
        .find(|setting| setting.name == option)
        .map(|setting| setting.values)
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

pub fn check(repository: &Repository) -> Result<(), String> {
    let root = repository.root.as_path();
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
    problems.extend(option_file_problems(repository)?);
    problems.extend(shader_path_problems(root)?);
    if problems.is_empty() {
        println!(
            "bench/ matches the code: {} catalog entries, the option files and shader paths.",
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

fn option_file_problems(repository: &Repository) -> Result<Vec<String>, String> {
    let root = repository.root.as_path();
    let mut problems = Vec::new();
    for (folder, fields) in [("scenes", SCENE_FIELDS), ("settings", SETTINGS_FIELDS)] {
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
            if let Some(field) = file
                .as_object()
                .into_iter()
                .flat_map(|fields| fields.keys())
                .find(|field| !fields.contains(&field.as_str()))
            {
                problems.push(format!(
                    "{at}: {field} is not a field ({})",
                    fields.join(", ")
                ));
            }
            let Some(options) = file["options"].as_object() else {
                problems.push(format!("{at}: options is an object of options by name"));
                continue;
            };
            for (option, value) in options {
                let value = switch(value);
                match setting_values(option) {
                    Some(values) if !values.contains(&value.as_str()) => problems.push(format!(
                        "{at}: {option} is one of {}, not {value}",
                        values.join(", ")
                    )),
                    Some(_) => {}
                    None if folder == "settings" && !HOST_OPTIONS.contains(&option.as_str()) => {
                        problems.push(format!(
                            "{at}: {option} is not a host setting ({} or an engine switch)",
                            HOST_OPTIONS.join(", ")
                        ));
                    }
                    None => {}
                }
            }
            if folder == "scenes" {
                match file["project"].as_str() {
                    Some(project) => {
                        let native = repository
                            .projects()
                            .find(|package| package.path.as_deref() == Some(project))
                            .map(|package| package.native().is_some());
                        match native {
                            Some(true) => {}
                            Some(false) => problems.push(format!(
                                "{at}: {project}/Cargo.toml: [package.metadata.engine] declares \
                                 no native"
                            )),
                            None => problems.push(format!(
                                "{at}: project {project} is no game crate's folder"
                            )),
                        }
                    }
                    None => problems.push(format!("{at}: project names a game crate's folder")),
                }
            }
        }
    }
    Ok(problems)
}

fn shader_path_problems(root: &Path) -> Result<Vec<String>, String> {
    let patterns: Vec<String> = serde_json::from_value(read(root, "shader-paths.json")?)
        .map_err(|error| format!("bench/shader-paths.json is a list of path patterns: {error}"))?;
    let files = crate::files::listed(root, &[])?;
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
