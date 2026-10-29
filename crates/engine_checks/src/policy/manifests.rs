use std::fs;
use std::path::Path;

use toml_edit::{Document, Item, TableLike};

use super::Problems;
use crate::Repository;

const DEPENDENCY_TABLES: &[&str] = &["dependencies", "dev-dependencies", "build-dependencies"];

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
        None if whole => problems.file(
            "Cargo.toml",
            "the workspace declares no [workspace.lints] table",
        ),
        None => {}
    }
    let inherited_keys: Vec<&str> = table(workspace, "package")
        .map(|package| package.iter().map(|(key, _)| key).collect())
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
        check_member(&path, manifest.as_item(), &inherited_keys, problems);
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
        match version {
            Some(version) if !version.starts_with('=') => problems.file(
                "Cargo.toml",
                &format!(
                    "[workspace.dependencies] {name} = {version:?} must pin its version with `=` \
                     (the version Cargo.lock holds)"
                ),
            ),
            None if !local => problems.file(
                "Cargo.toml",
                &format!(
                    "[workspace.dependencies] {name} names neither an exact version nor a path"
                ),
            ),
            _ => {}
        }
    }
}

fn check_member(path: &str, manifest: &Item, inherited_keys: &[&str], problems: &mut Problems) {
    let package = manifest.get("package");
    for key in inherited_keys {
        if !package
            .and_then(|package| package.get(key))
            .is_some_and(inherited)
        {
            problems.file(
                path,
                &format!(
                    "[package] must inherit `{key}` from the workspace (`{key}.workspace = true`)"
                ),
            );
        }
    }
    let lints = manifest.get("lints");
    match lints.and_then(Item::as_table_like) {
        Some(table) if lints.is_some_and(inherited) && table.len() == 1 => {}
        Some(_) => problems.file(
            path,
            "[lints] must be exactly `workspace = true`: one lint table, the workspace's, for \
             every crate (no local [lints.*] tables)",
        ),
        None => problems.file(
            path,
            "add `[lints] workspace = true`: every crate takes the workspace's lints",
        ),
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
                problems.file(
                    path,
                    &format!(
                        "[{kind}] {name} must come from the workspace (`{name}.workspace = true`, \
                         declared once in [workspace.dependencies] with an exact version)"
                    ),
                );
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
        problems.file(
            "rust-toolchain.toml",
            &format!("the toolchain channel {channel:?} must be an exact version (1.2.3)"),
        );
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
                problems.file(
                    path,
                    &format!("{kind} {name} {version:?} must be an exact version (1.2.3)"),
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
