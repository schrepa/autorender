//! The engine's features stay composable: each feature's folder in
//! engine_core (engine_core/src/features/NAME, named as `engine_features!`
//! names the feature) uses the core and the features it declares needing,
//! never another feature's types; the core uses none, but for extraction,
//! which hands the renderer what each feature drew, and the crate root,
//! which exports them. Folders and features match one to one, but for the
//! features in crates of their own.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use engine::Feature;

/// The crate of `feature` when it is not engine_core, whose features each
/// have a folder.
fn own_crate(feature: Feature) -> Option<&'static str> {
    match feature {
        Feature::Collision => Some("engine_collision"),
        Feature::Terrain
        | Feature::WorldStreaming
        | Feature::Navigation
        | Feature::Ocean
        | Feature::Animation
        | Feature::TimeOfDay
        | Feature::Deformation => None,
    }
}

/// Core modules that may name features' types: the crate root exporting
/// them, the sections list, and extraction handing renderers what each
/// feature draws.
const BRIDGES: &[&str] = &[
    "lib",
    "features",
    "environment",
    "retained_render",
    "scene",
    "frame_stats",
    "extraction",
];

/// engine_core's modules: each top-level module's name and its source text,
/// directories' files joined, comments and strings left out. Each folder of
/// features/ is a module of its own, `features/NAME`; the rest of features/
/// is the `features` module.
fn sources(root: &Path) -> Result<BTreeMap<String, String>, String> {
    let source = root.join("crates/engine_core/src");
    let mut modules = BTreeMap::new();
    for (name, path) in entries(&source)? {
        if name == "features" && path.is_dir() {
            for (feature, path) in entries(&path)? {
                let module = if path.is_dir() {
                    format!("features/{feature}")
                } else {
                    name.clone()
                };
                add_source(&mut modules, module, path)?;
            }
        } else {
            add_source(&mut modules, name, path)?;
        }
    }
    Ok(modules)
}

/// The entries of `directory`: each one's name without its extension, and
/// its path.
fn entries(directory: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(directory)
        .map_err(|error| format!("cannot read {}: {error}", directory.display()))?
    {
        let path = entry.map_err(|error| error.to_string())?.path();
        let name = path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_owned();
        entries.push((name, path));
    }
    Ok(entries)
}

/// Adds the code of `path` (a file, or a directory's files) to `module`'s.
fn add_source(
    modules: &mut BTreeMap<String, String>,
    module: String,
    path: PathBuf,
) -> Result<(), String> {
    let mut text = String::new();
    let mut files = vec![path];
    while let Some(file) = files.pop() {
        if file.is_dir() {
            for entry in fs::read_dir(&file).map_err(|error| error.to_string())? {
                files.push(entry.map_err(|error| error.to_string())?.path());
            }
        } else if file.extension().is_some_and(|extension| extension == "rs") {
            text.push_str(&code(
                &fs::read_to_string(&file).map_err(|error| error.to_string())?,
            ));
            text.push('\n');
        }
    }
    modules.entry(module).or_default().push_str(&text);
    Ok(())
}

/// `source` without comments and string literals.
fn code(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    for line in source.lines() {
        let mut in_string = false;
        let mut previous = ' ';
        let mut kept = String::new();
        let mut chars = line.chars().peekable();
        while let Some(c) = chars.next() {
            if !in_string && c == '/' && chars.peek() == Some(&'/') {
                break;
            }
            if c == '"' && previous != '\\' {
                in_string = !in_string;
            } else if !in_string {
                kept.push(c);
            }
            previous = c;
        }
        out.push_str(&kept);
        out.push('\n');
    }
    out
}

/// The identifiers of `code`.
fn identifiers(code: &str) -> BTreeSet<&str> {
    code.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|word| word.starts_with(|c: char| c.is_ascii_uppercase()))
        .collect()
}

/// The types `code` declares visible to the crate.
fn declared(code: &str) -> BTreeSet<String> {
    let mut types = BTreeSet::new();
    for line in code.lines() {
        let line = line.trim_start();
        let Some(rest) = line
            .strip_prefix("pub ")
            .or_else(|| line.strip_prefix("pub(crate) "))
        else {
            continue;
        };
        for keyword in ["struct ", "enum ", "trait ", "type "] {
            if let Some(name) = rest.strip_prefix(keyword) {
                let name: String = name
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() {
                    types.insert(name);
                }
            }
        }
    }
    types
}

/// Fails at the first module using a feature it may not, at a folder of
/// features/ that names no feature, at a feature of engine_core without a
/// folder, and at what [`listed`] finds outside its list.
pub fn check(root: &Path) -> Result<(), String> {
    let sources = sources(root)?;
    let owner = |module: &str| {
        module.strip_prefix("features/").and_then(|name| {
            Feature::ALL
                .into_iter()
                .find(|feature| feature.name() == name)
        })
    };
    let mut violations = Vec::new();
    listed(root, &mut violations)?;
    for module in sources.keys() {
        if module.starts_with("features/") && owner(module).is_none() {
            violations.push(format!(
                "{module} is no feature's folder: name it as engine_features! names one"
            ));
        }
    }
    for feature in Feature::ALL {
        let folder = sources.contains_key(&format!("features/{}", feature.name()));
        match own_crate(feature) {
            None if !folder => violations.push(format!(
                "{} has no folder: its code is engine_core/src/features/{}",
                feature.name(),
                feature.name()
            )),
            Some(home) if folder => violations.push(format!(
                "{} lives in {home} and has a folder in engine_core too",
                feature.name()
            )),
            None | Some(_) => {}
        }
    }
    let mut types: BTreeMap<String, Feature> = BTreeMap::new();
    for (module, code) in &sources {
        if let Some(feature) = owner(module) {
            for name in declared(code) {
                types.insert(name, feature);
            }
        }
    }
    for (module, code) in &sources {
        if BRIDGES.contains(&module.as_str()) {
            continue;
        }
        let own = owner(module);
        let allowed = own.map_or(&[][..], Feature::needs);
        let mut used: BTreeSet<String> = BTreeSet::new();
        for word in identifiers(code) {
            if let Some(&feature) = types.get(word)
                && Some(feature) != own
                && !allowed.contains(&feature)
            {
                used.insert(format!("{word} ({})", feature.name()));
            }
        }
        if !used.is_empty() {
            violations.push(format!(
                "{module} ({}) uses {}",
                own.map_or("core", Feature::name),
                used.into_iter().collect::<Vec<_>>().join(", ")
            ));
        }
    }
    if violations.is_empty() {
        return Ok(());
    }
    Err(format!(
        "engine features must stay composable and listed: a feature uses the core and the \
         features it needs (engine_features! in engine_core/src/features/mod.rs; its code is \
         its folder there), the core no feature, and each feature's development file and \
         queries are in their lists:\n{}",
        violations.join("\n")
    ))
}

/// Adds to `violations` each development file of a feature that
/// `dev_features!` (engine_dev/src/features/mod.rs) does not list, whose
/// inspectors and schemas nothing would register, each query a feature
/// answers (`impl EngineQuery for T`) or control it applies (`impl
/// EngineControl for T`) that no `DevRequest` or `BrowserControl` variant
/// holds, which no request would reach, and each scene component (`impl
/// SceneComponent for T`) that `entity_components!`
/// (engine_scene/src/components.rs) does not list, which no scene would
/// author.
fn listed(root: &Path, violations: &mut Vec<String>) -> Result<(), String> {
    let folder = root.join("crates/engine_dev/src/features");
    let read = |path: &Path| {
        fs::read_to_string(path)
            .map(|text| code(&text))
            .map_err(|error| format!("cannot read {}: {error}", path.display()))
    };
    let list = read(&folder.join("mod.rs"))?;
    let names: Vec<&str> = list
        .split_once("macro_rules! dev_features")
        .and_then(|(_, rest)| rest.split_once("$then!"))
        .and_then(|(_, rest)| rest.split_once('{'))
        .and_then(|(_, rest)| rest.split_once('}'))
        .map(|(body, _)| body.split(',').map(str::trim).collect())
        .ok_or("engine_dev/src/features/mod.rs declares no dev_features! list")?;
    let protocol = read(&root.join("crates/engine_dev/src/protocol.rs"))?;
    for (name, path) in entries(&folder)? {
        if name == "mod" {
            continue;
        }
        if !names.contains(&name.as_str()) {
            violations.push(format!(
                "engine_dev/src/features/{name} is in no dev_features! entry, so nothing \
                 registers its inspectors: list {name} there"
            ));
        }
        let mut files = vec![path];
        while let Some(file) = files.pop() {
            if file.is_dir() {
                files.extend(entries(&file)?.into_iter().map(|(_, path)| path));
                continue;
            }
            let text = read(&file)?;
            // `impl EngineQuery for T` and `impl EngineControl for T`, their
            // trait's path written or not: a protocol variant holds T.
            for (kind, request) in [("query", "DevRequest"), ("control", "BrowserControl")] {
                let pattern = if kind == "query" {
                    "EngineQuery for "
                } else {
                    "EngineControl for "
                };
                for (at, _) in text.match_indices(pattern) {
                    let held: String = text[at + pattern.len()..]
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    if !protocol.contains(&format!("::{held})"))
                        && !protocol.contains(&format!("({held})"))
                    {
                        violations.push(format!(
                            "{name}'s {kind} {held} is held by no {request} variant: add one"
                        ));
                    }
                }
            }
        }
    }
    let scene = root.join("crates/engine_scene/src");
    let components = read(&scene.join("components.rs"))?;
    let list = components
        .split_once("macro_rules! entity_components")
        .and_then(|(_, rest)| rest.split_once("$then!"))
        .map(|(_, rest)| rest)
        .ok_or("engine_scene/src/components.rs declares no entity_components! list")?;
    // A type as entries and impls name it: its path's last segment.
    let last = |path: &str| path.rsplit("::").next().unwrap_or(path).trim().to_owned();
    // Each entry's type, `field: Type {`.
    let listed: Vec<String> = list
        .lines()
        .filter_map(|line| line.trim().split_once(": "))
        .filter(|(field, _)| field.chars().all(|c| c.is_ascii_lowercase() || c == '_'))
        .filter_map(|(_, rest)| rest.split_once(" {"))
        .map(|(path, _)| last(path))
        .collect();
    for (_, path) in entries(&scene)? {
        let text = read(&path)?;
        for (at, _) in text.match_indices("SceneComponent for ") {
            let rest = &text[at + "SceneComponent for ".len()..];
            let component = last(rest.split_once(" {").map_or(rest, |(path, _)| path));
            if !component.starts_with('$') && !listed.contains(&component) {
                violations.push(format!(
                    "scene component {component} is in no entity_components! entry: list it"
                ));
            }
        }
    }
    Ok(())
}
