//! The hash of the source code a crate builds from, for the crates whose
//! code generates records hosts keep and the cook ships (terrain tiles,
//! their bakes): a build script calls [`emit`], and the crate includes
//! `source.rs` from its build's output folder, which declares
//! `pub const SOURCE: [u8; 32]`. Records keyed by it stay valid across
//! builds that leave that code alone, and none outlives a change to it:
//! nobody bumps a version.
//!
//! The hash covers every file of the crate and of the crates of this
//! repository it depends on, transitively (its `dependencies` tables, as
//! the workspace declares their paths), the sources the workspace patches in
//! for registry crates any of them depends on, the lockfile (every
//! registry crate's version) and the toolchain file. Line endings are
//! hashed as git holds them, so every checkout hashes the same; the
//! target never is, as records are the same bits on every host.
//!
//! A crate keying records whose code it does not link (records only a
//! tool makes, such as the cook's levels of detail, which runtimes load)
//! hashes that code's crate instead ([`emit_of`]).
//!
//! [`record_key`] is the key records are kept by, from that hash.
#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// The key hosts keep a record by, in their stores and the cook's packs:
/// SHA-256 over `source`, the hash of the code making it, its `kind` and
/// its `identity`, everything its bytes depend on.
pub fn record_key(source: &[u8; 32], kind: &str, identity: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(source);
    hash.update(kind.as_bytes());
    hash.update([0]);
    hash.update(identity);
    hash.finalize().into()
}

/// Hashes the calling crate's sources and writes `source.rs` into the
/// build's output folder, telling Cargo to run the build script again when
/// any file hashed changes. Panics, failing the build, where a manifest or
/// a file cannot be read.
pub fn emit() {
    let manifest = PathBuf::from(var("CARGO_MANIFEST_DIR"));
    write_source(&manifest, "source.rs", "this crate");
}

/// [`emit`] for the workspace's crate `name` rather than the calling one,
/// for a crate keying records `name`'s code makes without linking it:
/// writes `{name}_source.rs`, declaring `SOURCE` as `name`'s own build
/// would. Panics, failing the build, where the workspace names no such
/// crate.
pub fn emit_of(name: &str) {
    let manifest = PathBuf::from(var("CARGO_MANIFEST_DIR"));
    let root = workspace_root(&manifest);
    let workspace = read(&root.join("Cargo.toml"));
    let paths = table_paths(&workspace, "workspace.dependencies");
    let folder = paths
        .get(name)
        .map(|path| root.join(path))
        .unwrap_or_else(|| panic!("the workspace names no crate {name}"));
    write_source(
        &folder,
        &format!("{name}_source.rs"),
        &format!("the crate `{name}`"),
    );
}

/// Hashes the crate in `manifest` (its folder) as [`emit`] describes, and
/// writes `SOURCE` into the build's output folder as `file`, its doc naming
/// the crate as `crate_name`.
fn write_source(manifest: &Path, file: &str, crate_name: &str) {
    let root = workspace_root(manifest);
    let workspace = read(&root.join("Cargo.toml"));
    let paths = table_paths(&workspace, "workspace.dependencies");
    let patches = table_paths(&workspace, "patch.crates-io");
    let mut folders = BTreeSet::new();
    let mut queue = vec![manifest.to_path_buf()];
    while let Some(folder) = queue.pop() {
        if !folders.insert(folder.clone()) {
            continue;
        }
        let crate_manifest = read(&folder.join("Cargo.toml"));
        for name in dependency_names(&crate_manifest) {
            if let Some(path) = paths.get(&name) {
                queue.push(root.join(path));
            } else if let Some(path) = patches.get(&name) {
                folders.insert(root.join(path));
            }
        }
    }
    let mut files = Vec::new();
    for folder in &folders {
        println!("cargo:rerun-if-changed={}", folder.display());
        list(folder, &mut files);
    }
    for file in ["Cargo.lock", "rust-toolchain.toml"] {
        let path = root.join(file);
        println!("cargo:rerun-if-changed={}", path.display());
        files.push(path);
    }
    let mut named: Vec<(String, PathBuf)> = files
        .into_iter()
        .map(|path| {
            let relative = path
                .strip_prefix(&root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            (relative, path)
        })
        .collect();
    named.sort();
    let mut hash = Sha256::new();
    for (name, path) in &named {
        let bytes = fs::read(path).unwrap_or_else(|error| panic!("cannot read {name}: {error}"));
        hash.update(name.as_bytes());
        hash.update([0]);
        hash.update(normalized(&bytes));
        hash.update([0]);
    }
    let digest = hash.finalize();
    let out = PathBuf::from(var("OUT_DIR")).join(file);
    let bytes: Vec<String> = digest.iter().map(|byte| format!("{byte:#04x}")).collect();
    let source = format!(
        "/// The hash of the source code {crate_name} builds from: its own, what\n\
         /// it depends on in its repository, the lockfile and the toolchain.\n\
         pub const SOURCE: [u8; 32] = [{}];\n",
        bytes.join(", ")
    );
    fs::write(&out, source)
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", out.display()));
}

fn var(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set: run from a build script"))
}

fn read(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}

/// The folder holding the workspace's manifest: the nearest above
/// `manifest` declaring `[workspace]`.
fn workspace_root(manifest: &Path) -> PathBuf {
    manifest
        .ancestors()
        .find(|folder| {
            fs::read_to_string(folder.join("Cargo.toml"))
                .is_ok_and(|text| text.lines().any(|line| line.trim() == "[workspace]"))
        })
        .unwrap_or_else(|| panic!("no workspace above {}", manifest.display()))
        .to_path_buf()
}

/// The paths a manifest's `table` gives its entries (`name = { path = "..." }`),
/// by name.
fn table_paths(manifest: &str, table: &str) -> BTreeMap<String, String> {
    let mut paths = BTreeMap::new();
    let mut inside = false;
    for line in manifest.lines() {
        let line = line.trim();
        if let Some(header) = line.strip_prefix('[') {
            inside = header.trim_end_matches(']').trim() == table;
            continue;
        }
        if !inside {
            continue;
        }
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        if let Some(path) = quoted_after(value, "path") {
            paths.insert(name.trim().trim_matches('"').to_owned(), path);
        }
    }
    paths
}

/// The names of the dependencies a crate's manifest declares in its
/// `[dependencies]` tables, its targets' included: the policy has every
/// one inherited from the workspace (`name.workspace = true` or
/// `name = { workspace = true, ... }`).
fn dependency_names(manifest: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut inside = false;
    for line in manifest.lines() {
        let line = line.trim();
        if let Some(header) = line.strip_prefix('[') {
            let header = header.trim_end_matches(']').trim();
            inside = header == "dependencies" || header.ends_with(".dependencies");
            inside &=
                !header.ends_with("dev-dependencies") && !header.ends_with("build-dependencies");
            continue;
        }
        if !inside || line.starts_with('#') {
            continue;
        }
        let Some((key, _)) = line.split_once('=') else {
            continue;
        };
        let name = key.trim().trim_end_matches(".workspace").trim_matches('"');
        if !name.is_empty() {
            names.push(name.to_owned());
        }
    }
    names
}

/// The quoted string `value` gives `key` (`key = "..."`).
fn quoted_after(value: &str, key: &str) -> Option<String> {
    let at = value.find(key)?;
    let rest = value[at + key.len()..].trim_start().strip_prefix('=')?;
    let rest = rest.trim_start().strip_prefix('"')?;
    rest.split_once('"').map(|(path, _)| path.to_owned())
}

/// Every file under `folder` a checkout holds: not build outputs, hidden
/// files or editors' scratch files, which differ between machines.
fn list(folder: &Path, files: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(folder)
        .unwrap_or_else(|error| panic!("cannot list {}: {error}", folder.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|error| panic!("cannot list {}: {error}", folder.display()))
            .path();
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let scratch = name.starts_with('.')
            || name.ends_with('~')
            || [".swp", ".tmp", ".orig", ".rej"]
                .iter()
                .any(|suffix| name.ends_with(suffix));
        if scratch {
            continue;
        }
        if path.is_dir() {
            if name != "target" {
                list(&path, files);
            }
        } else {
            files.push(path);
        }
    }
}

/// `bytes` with Windows line endings as Unix ones.
fn normalized(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    for (index, &byte) in bytes.iter().enumerate() {
        if !(byte == b'\r' && bytes.get(index + 1) == Some(&b'\n')) {
            out.push(byte);
        }
    }
    out
}
