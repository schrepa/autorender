//! The engine's features stay composable: each feature's modules in
//! engine_core use the core and the features they declare needing, never
//! another feature's types; the core uses none, but for extraction, which
//! hands the renderer what each feature drew, and the crate root, which
//! exports them.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use engine::Feature;

/// `feature`'s modules (files and directories of engine_core/src); its name
/// and the features it may use come from the engine's one list of features
/// (`engine_features!`). Collision lives in engine_collision.
fn modules(feature: Feature) -> &'static [&'static str] {
    match feature {
        Feature::Terrain => &[
            "terrain",
            "bridges",
            "waterfalls",
            "water",
            "ground_cover",
            "wind",
            "occupancy",
        ],
        Feature::WorldStreaming => &["streaming"],
        Feature::Navigation => &["navigation"],
        Feature::Ocean => &["ocean"],
        Feature::Animation => &["animation", "skin"],
        Feature::TimeOfDay => &["time_of_day"],
        Feature::Deformation => &["deformation"],
        Feature::Collision => &[],
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
/// directories' files joined, comments and strings left out.
fn sources(root: &Path) -> Result<BTreeMap<String, String>, String> {
    let source = root.join("crates/engine_core/src");
    let mut modules = BTreeMap::new();
    for entry in
        fs::read_dir(&source).map_err(|error| format!("cannot read engine_core: {error}"))?
    {
        let path = entry.map_err(|error| error.to_string())?.path();
        let name = path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_owned();
        let mut text = String::new();
        let mut files = vec![path.clone()];
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
        modules
            .entry(name)
            .or_insert_with(String::new)
            .push_str(&text);
    }
    Ok(modules)
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

/// Fails at the first module using a feature it may not.
pub fn check(root: &Path) -> Result<(), String> {
    let sources = sources(root)?;
    let owner = |module: &str| {
        Feature::ALL
            .into_iter()
            .find(|&feature| modules(feature).contains(&module))
    };
    let mut types: BTreeMap<String, Feature> = BTreeMap::new();
    for (module, code) in &sources {
        if let Some(feature) = owner(module) {
            for name in declared(code) {
                types.insert(name, feature);
            }
        }
    }
    let mut violations = Vec::new();
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
        "engine features must stay composable: a feature uses the core and the features it \
         needs (engine_features! in engine_core/src/features/mod.rs; its modules are in \
         tools/xtask/src/features.rs), the core no feature:\n{}",
        violations.join("\n")
    ))
}
