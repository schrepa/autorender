//! Manifest rules: one lint table for every crate, package keys and
//! dependencies inherited from the workspace, registry dependencies pinned
//! exactly, an exact toolchain and exact web dependency versions.

use std::fs;
use std::path::Path;

use toml_edit::{Document, Item, TableLike};

use crate::Problems;

/// The dependency tables of a manifest (each also under `[target.*]`).
const DEPENDENCY_TABLES: &[&str] = &["dependencies", "dev-dependencies", "build-dependencies"];

/// The web manifests' dependency tables.
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

/// Checks the workspace manifest, every member's, the toolchain file and
/// the web manifests in `package_jsons`.
pub fn check(root: &Path, package_jsons: &[String], problems: &mut Problems) -> Result<(), String> {
    let workspace_manifest = parse(root, "Cargo.toml")?;
    let workspace = workspace_manifest
        .as_item()
        .get("workspace")
        .ok_or("Cargo.toml has no [workspace]")?;
    if table(workspace, "lints").is_none() {
        problems.file(
            "Cargo.toml",
            "the workspace declares no [workspace.lints] table",
        );
    }
    let inherited_keys: Vec<&str> = table(workspace, "package")
        .map(|package| package.iter().map(|(key, _)| key).collect())
        .unwrap_or_default();
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
    for member in members(root, workspace)? {
        let path = format!("{member}/Cargo.toml");
        let manifest = parse(root, &path)?;
        check_member(&path, manifest.as_item(), &inherited_keys, problems);
    }
    check_toolchain(root, problems)?;
    for path in package_jsons {
        check_package_json(root, path, problems)?;
    }
    Ok(())
}

/// The workspace's members, `dir/*` patterns expanded to the folders under
/// `dir` holding a Cargo.toml.
fn members(root: &Path, workspace: &Item) -> Result<Vec<String>, String> {
    let mut members = Vec::new();
    let patterns = workspace
        .get("members")
        .and_then(Item::as_array)
        .ok_or("Cargo.toml lists no [workspace] members")?;
    for pattern in patterns.iter().filter_map(|pattern| pattern.as_str()) {
        let Some(parent) = pattern.strip_suffix("/*") else {
            if pattern.contains(['*', '?', '[']) {
                return Err(format!(
                    "tools/policy reads member patterns `dir/*` only, not {pattern:?}"
                ));
            }
            members.push(pattern.to_owned());
            continue;
        };
        let mut found = Vec::new();
        for entry in fs::read_dir(root.join(parent))
            .map_err(|error| format!("cannot read the members under {parent}: {error}"))?
        {
            let entry = entry.map_err(|error| error.to_string())?;
            if entry.path().join("Cargo.toml").is_file() {
                found.push(format!("{parent}/{}", entry.file_name().to_string_lossy()));
            }
        }
        found.sort();
        members.extend(found);
    }
    Ok(members)
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

/// Whether `version` is one exact version: `1.2.3`, a pre-release allowed.
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
