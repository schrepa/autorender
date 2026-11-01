use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use engine_core::{Error, ErrorCode};
use serde_json::json;
use toml_edit::{Document, Item, TableLike};

use super::Problems;
use crate::Repository;

const DEPENDENCY_TABLES: &[&str] = &["dependencies", "dev-dependencies", "build-dependencies"];

const UNSAFE_CODE: &str = "rust::unsafe_code";

const PACKAGE_JSON_TABLES: &[&str] = &[
    "dependencies",
    "devDependencies",
    "optionalDependencies",
    "peerDependencies",
];

fn parse(root: &Path, path: &str) -> Result<Document<String>, String> {
    let text = fs::read_to_string(root.join(path))
        .map_err(|error| format!("cannot read {path}: {error}"))?;
    Document::parse(text).map_err(|error| format!("{path} is not TOML: {error}"))
}

fn table<'a>(item: &'a Item, key: &str) -> Option<&'a dyn TableLike> {
    item.get(key).and_then(Item::as_table_like)
}

fn inherited(item: &Item) -> bool {
    item.as_table_like()
        .and_then(|table| table.get("workspace"))
        .and_then(Item::as_bool)
        == Some(true)
}

pub fn check(
    repository: &Repository,
    package_jsons: &[String],
    problems: &mut Problems,
) -> Result<Vec<String>, String> {
    let root = repository.root.as_path();
    let workspace_folder = repository.workspace_root.as_path();
    let workspace_manifest = parse(workspace_folder, "Cargo.toml")?;
    let workspace = workspace_manifest
        .as_item()
        .get("workspace")
        .ok_or("Cargo.toml has no [workspace]")?;
    let whole = repository.is_workspace_root();
    let mut lints = Vec::new();
    match table(workspace, "lints") {
        Some(tools) => {
            for (tool, table) in tools.iter() {
                for (lint, _) in table.as_table_like().into_iter().flat_map(TableLike::iter) {
                    lints.push(match tool {
                        "rust" => lint.to_owned(),
                        _ => format!("{tool}::{lint}"),
                    });
                }
            }
        }
        None if whole => {
            problems.push(Error::new(ErrorCode::LintMissing, "Cargo.toml").got("workspace.lints"));
        }
        None => {}
    }
    let inherited_keys: Vec<&str> = table(workspace, "package")
        .map(|package| package.iter().map(|(key, _)| key).collect())
        .unwrap_or_default();
    let levels = table(workspace, "lints")
        .map(lint_levels)
        .unwrap_or_default();
    if whole {
        pinned(workspace, problems);
        check_toolchain(root, problems)?;
    }
    for member in repository.own() {
        let Some(path) = member.file("Cargo.toml") else {
            continue;
        };
        let manifest = parse(root, &path)?;
        check_member(
            &path,
            manifest.as_item(),
            &inherited_keys,
            &levels,
            problems,
        );
    }
    for path in package_jsons {
        check_package_json(root, path, problems)?;
    }
    Ok(lints)
}

fn pinned(workspace: &Item, problems: &mut Problems) {
    for (name, dependency) in table(workspace, "dependencies")
        .into_iter()
        .flat_map(TableLike::iter)
    {
        let version = dependency.as_str().or_else(|| {
            dependency
                .as_table_like()
                .and_then(|table| table.get("version"))
                .and_then(Item::as_str)
        });
        let local = dependency
            .as_table_like()
            .is_some_and(|table| table.contains_key("path"));
        let unpinned = match version {
            Some(version) if !version.starts_with('=') => Some(json!({ name: version })),
            None if !local => Some(json!({ name: null })),
            _ => None,
        };
        if let Some(unpinned) = unpinned {
            problems.push(Error::new(ErrorCode::VersionUnpinned, "Cargo.toml").got(unpinned));
        }
    }
}

fn lint_levels(lints: &dyn TableLike) -> BTreeMap<String, String> {
    let mut levels = BTreeMap::new();
    for (tool, table) in lints.iter() {
        for (lint, level) in table.as_table_like().into_iter().flat_map(TableLike::iter) {
            let level = level
                .get("level")
                .unwrap_or(level)
                .as_str()
                .unwrap_or_default();
            levels.insert(format!("{tool}::{lint}"), level.to_owned());
        }
    }
    levels
}

fn lowers_unsafe_only(lints: &dyn TableLike, levels: &BTreeMap<String, String>) -> bool {
    let mut local = lint_levels(lints);
    let lowered = local.remove(UNSAFE_CODE);
    let mut shared = levels.clone();
    shared.remove(UNSAFE_CODE);
    local == shared && lowered.as_deref() == Some("deny")
}

fn check_member(
    path: &str,
    manifest: &Item,
    inherited_keys: &[&str],
    levels: &BTreeMap<String, String>,
    problems: &mut Problems,
) {
    let package = manifest.get("package");
    for key in inherited_keys {
        if !package
            .and_then(|package| package.get(key))
            .is_some_and(inherited)
        {
            problems.push(Error::new(ErrorCode::NotInherited, path).got(format!("package.{key}")));
        }
    }
    let lints = manifest.get("lints");
    match lints.and_then(Item::as_table_like) {
        Some(table) if lints.is_some_and(inherited) && table.len() == 1 => {}
        Some(table) if lowers_unsafe_only(table, levels) => {}
        Some(_) => problems.push(Error::new(ErrorCode::NotInherited, path).got("lints")),
        None => problems.push(Error::new(ErrorCode::LintMissing, path).got("lints")),
    }
    let mut tables: Vec<(String, &dyn TableLike)> = Vec::new();
    for kind in DEPENDENCY_TABLES {
        if let Some(dependencies) = table(manifest, kind) {
            tables.push(((*kind).to_owned(), dependencies));
        }
    }
    for (platform, target) in table(manifest, "target")
        .into_iter()
        .flat_map(TableLike::iter)
    {
        for kind in DEPENDENCY_TABLES {
            if let Some(dependencies) = target.get(kind).and_then(Item::as_table_like) {
                tables.push((format!("target.'{platform}'.{kind}"), dependencies));
            }
        }
    }
    for (kind, dependencies) in tables {
        for (name, dependency) in dependencies.iter() {
            if !inherited(dependency) {
                problems
                    .push(Error::new(ErrorCode::NotInherited, path).got(format!("{kind}.{name}")));
            }
        }
    }
}

fn check_toolchain(root: &Path, problems: &mut Problems) -> Result<(), String> {
    let toolchain = parse(root, "rust-toolchain.toml")?;
    let channel = toolchain
        .as_item()
        .get("toolchain")
        .and_then(|toolchain| toolchain.get("channel"))
        .and_then(Item::as_str)
        .unwrap_or_default();
    if !exact_version(channel) {
        problems.push(Error::new(ErrorCode::VersionUnpinned, "rust-toolchain.toml").got(channel));
    }
    Ok(())
}

fn check_package_json(root: &Path, path: &str, problems: &mut Problems) -> Result<(), String> {
    let text = fs::read_to_string(root.join(path))
        .map_err(|error| format!("cannot read {path}: {error}"))?;
    let manifest: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{path} is not JSON: {error}"))?;
    for kind in PACKAGE_JSON_TABLES {
        for (name, version) in manifest[kind].as_object().into_iter().flatten() {
            let version = version.as_str().unwrap_or_default();
            if !exact_version(version) {
                problems.push(
                    Error::new(ErrorCode::VersionUnpinned, path)
                        .got(json!({ format!("{kind}.{name}"): version })),
                );
            }
        }
    }
    Ok(())
}

fn exact_version(version: &str) -> bool {
    let (release, pre) = version.split_once('-').unwrap_or((version, ""));
    let parts: Vec<&str> = release.split('.').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
        && pre
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'.' || byte == b'-')
}
