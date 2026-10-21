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
//! for registry crates any of them depends on, the lockfile's entries of
//! the crates it builds with (each registry crate's version) and the
//! toolchain file. Line endings are hashed as git holds them, so every
//! checkout hashes the same; the target never is, as records are the same
//! bits on every host.
//!
//! A crate whose records come from parts of its code keys each kind by the
//! code making it alone ([`emit_sets`]): a [`Set`] of its top modules, the
//! modules they use followed through its code, so a change elsewhere in the
//! crate leaves those records as they were.
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

/// A kind of a crate's records keyed by the code making it alone
/// ([`emit_sets`]): `sources::NAME`, the hash of the crate's top modules
/// `roots`, the modules of the crate they use, followed through its code
/// but for those of `stop` (shared plumbing whose changes leave the
/// records' bytes alone, neither hashed nor followed), `files` (paths
/// within the crate's folder; one ending in `/` names every file under
/// it), the crate's manifest, build script and root file, and the
/// workspace crates `crates` names (`None`: every one it depends on, as
/// [`emit`] hashes), with the lockfile's entries and the toolchain file.
///
/// A module uses what its code reaches through the crate's root, as code
/// of this edition must name it: each name written after `crate::` or
/// `super::` (a top module's `super` is the root) and each first name of a
/// group after them (`use crate::{a, b::c}`: `a` and `b`), a top module or
/// an item the root re-exports from one (every module the root re-exports
/// whole, for an item it does not name): its code with comments and
/// strings left out, a name that only might be another module's followed
/// too. A module's files are its own file, everything under its folder and
/// the files it includes (`include_str!` and the like, by literal path).
#[derive(Clone, Copy, Debug)]
pub struct Set<'a> {
    pub name: &'a str,
    pub roots: &'a [&'a str],
    pub stop: &'a [&'a str],
    pub files: &'a [&'a str],
    pub crates: Option<&'a [&'a str]>,
}

/// Hashes the calling crate's sources and writes `source.rs` into the
/// build's output folder, telling Cargo to run the build script again when
/// any file hashed changes. Panics, failing the build, where a manifest or
/// a file cannot be read.
pub fn emit() {
    emit_sets(&[]);
}

/// [`emit`], and in `source.rs` a module `sources` declaring each of
/// `sets`' hashes by its name ([`Set`]), which the crate's records of that
/// kind are keyed by. Panics, failing the build, where a set names a
/// module the crate's root does not declare.
pub fn emit_sets(sets: &[Set<'_>]) {
    let manifest = PathBuf::from(var("CARGO_MANIFEST_DIR"));
    let mut source = source_declaration(&manifest, "this crate");
    if !sets.is_empty() {
        let modules = Modules::of(&manifest);
        source.push_str(
            "/// The hashes of the code making each kind of this crate's records:\n\
             /// the modules making it and what they use ([`engine_sources::Set`]).\n\
             pub mod sources {\n",
        );
        for set in sets {
            let digest = set_digest(&manifest, &modules, set);
            source.push_str(&format!(
                "    /// The hash of the code of {}, what it uses and what it builds with.\n\
                 \x20   pub const {}: [u8; 32] = [{}];\n",
                set.roots
                    .iter()
                    .map(|root| format!("`{root}`"))
                    .collect::<Vec<_>>()
                    .join(", "),
                set.name,
                bytes(&digest),
            ));
        }
        source.push_str("}\n");
    }
    write_out("source.rs", &source);
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
    let source = source_declaration(&folder, &format!("the crate `{name}`"));
    write_out(&format!("{name}_source.rs"), &source);
}

/// `SOURCE`'s declaration for the crate in `manifest` (its folder), its
/// doc naming the crate as `crate_name`: the hash of its files, those of
/// the workspace crates it depends on and the patched sources they build
/// with, its lockfile entries and the toolchain file.
fn source_declaration(manifest: &Path, crate_name: &str) -> String {
    let root = workspace_root(manifest);
    let mut files = Vec::new();
    for folder in &crate_folders(&root, manifest, None) {
        println!("cargo:rerun-if-changed={}", folder.display());
        list(folder, &mut files);
    }
    let digest = digest(&root, manifest, files);
    format!(
        "/// The hash of the source code {crate_name} builds from: its own, what\n\
         /// it depends on in its repository, its lockfile entries and the\n\
         /// toolchain.\n\
         pub const SOURCE: [u8; 32] = [{}];\n",
        bytes(&digest)
    )
}

/// The hash of `set` of the crate in `manifest`, whose modules are
/// `modules` ([`Set`]).
fn set_digest(manifest: &Path, modules: &Modules, set: &Set<'_>) -> [u8; 32] {
    let root = workspace_root(manifest);
    let mut files = Vec::new();
    for name in modules.closure(set.roots, set.stop) {
        files.extend(modules.files[&name].iter().cloned());
    }
    files.push(modules.lib.clone());
    for file in ["Cargo.toml", "build.rs"] {
        let path = manifest.join(file);
        if path.exists() {
            files.push(path);
        }
    }
    for file in set.files {
        let path = manifest.join(file);
        if file.ends_with('/') {
            list(&path, &mut files);
        } else {
            files.push(path);
        }
    }
    for folder in crate_folders(&root, manifest, set.crates) {
        if folder != manifest {
            list(&folder, &mut files);
        }
    }
    digest(&root, manifest, files)
}

/// SHA-256 over `files` (each by its path within the workspace at `root`,
/// once, in order of those paths), the lockfile's entries of the crate in
/// `manifest` and the toolchain file, the build asked to run again when
/// either changes.
fn digest(root: &Path, manifest: &Path, files: Vec<PathBuf>) -> [u8; 32] {
    let mut named: BTreeMap<String, PathBuf> = files
        .into_iter()
        .map(|path| {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            (relative, path)
        })
        .collect();
    let toolchain = root.join("rust-toolchain.toml");
    println!("cargo:rerun-if-changed={}", toolchain.display());
    named.insert("rust-toolchain.toml".to_owned(), toolchain);
    let mut hash = Sha256::new();
    for (name, path) in &named {
        let bytes = fs::read(path).unwrap_or_else(|error| panic!("cannot read {name}: {error}"));
        hash.update(name.as_bytes());
        hash.update([0]);
        hash.update(normalized(&bytes));
        hash.update([0]);
    }
    let lock = root.join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    let package = package_name(&read(&manifest.join("Cargo.toml")));
    hash.update(b"Cargo.lock");
    hash.update([0]);
    hash.update(lock_entries(&read(&lock), &package).as_bytes());
    hash.finalize().into()
}

/// The folders of the crate in `manifest` and of the workspace crates it
/// depends on, transitively, or of those `only` names (with the crate's
/// own), and the sources the workspace patches in for registry crates any
/// of them depends on.
fn crate_folders(root: &Path, manifest: &Path, only: Option<&[&str]>) -> BTreeSet<PathBuf> {
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
                if only.is_none_or(|only| only.contains(&name.as_str())) {
                    queue.push(root.join(path));
                }
            } else if let Some(path) = patches.get(&name) {
                folders.insert(root.join(path));
            }
        }
    }
    folders
}

/// The `[[package]]` entries of `lock` (a lockfile) of `package` and of
/// every crate it builds with, transitively, in the lockfile's order: whole
/// where it holds no entry of that name.
fn lock_entries(lock: &str, package: &str) -> String {
    let entries: Vec<LockEntry> = lock
        .split("[[package]]")
        .skip(1)
        .map(LockEntry::of)
        .collect();
    if !entries.iter().any(|entry| entry.name == package) {
        return lock.to_owned();
    }
    let mut taken = vec![false; entries.len()];
    let mut queue: Vec<(String, Option<String>)> = vec![(package.to_owned(), None)];
    while let Some((name, version)) = queue.pop() {
        for (index, entry) in entries.iter().enumerate() {
            let matches = entry.name == name
                && version
                    .as_ref()
                    .is_none_or(|version| *version == entry.version);
            if !matches || taken[index] {
                continue;
            }
            taken[index] = true;
            for dependency in &entry.dependencies {
                let mut parts = dependency.split_whitespace();
                let name = parts.next().unwrap_or_default().to_owned();
                let version = parts.next().map(str::to_owned);
                queue.push((name, version));
            }
        }
    }
    entries
        .iter()
        .zip(taken)
        .filter(|(_, taken)| *taken)
        .map(|(entry, _)| entry.text.as_str())
        .collect::<Vec<_>>()
        .join("[[package]]")
}

/// A lockfile's `[[package]]` entry: its text, name, version and the
/// crates it depends on (`name` or `name version`, as the lockfile writes
/// them).
struct LockEntry {
    text: String,
    name: String,
    version: String,
    dependencies: Vec<String>,
}

impl LockEntry {
    fn of(text: &str) -> Self {
        let mut entry = Self {
            text: text.replace("\r\n", "\n"),
            name: String::new(),
            version: String::new(),
            dependencies: Vec::new(),
        };
        let mut listing = false;
        for line in text.lines() {
            let line = line.trim();
            if listing {
                if line.starts_with(']') {
                    listing = false;
                } else if let Some(quoted) = line.trim_end_matches(',').strip_prefix('"') {
                    entry
                        .dependencies
                        .push(quoted.trim_end_matches('"').to_owned());
                }
                continue;
            }
            if let Some(value) = line.strip_prefix("name = ") {
                entry.name = value.trim_matches('"').to_owned();
            } else if let Some(value) = line.strip_prefix("version = ") {
                entry.version = value.trim_matches('"').to_owned();
            } else if line.starts_with("dependencies = [") {
                listing = !line.ends_with(']');
            }
        }
        entry
    }
}

/// The name a crate's manifest gives its package.
fn package_name(manifest: &str) -> String {
    let mut inside = false;
    for line in manifest.lines() {
        let line = line.trim();
        if let Some(header) = line.strip_prefix('[') {
            inside = header.trim_end_matches(']').trim() == "package";
            continue;
        }
        if inside
            && let Some((key, value)) = line.split_once('=')
            && key.trim() == "name"
        {
            return value.trim().trim_matches('"').to_owned();
        }
    }
    String::new()
}

/// A crate's top modules, as its root file declares them: each one's
/// files and the names its code reaches through the root; the items the
/// root re-exports, by name, with the modules each comes from, and the
/// modules it re-exports whole.
struct Modules {
    lib: PathBuf,
    files: BTreeMap<String, Vec<PathBuf>>,
    names: BTreeMap<String, BTreeSet<String>>,
    exports: BTreeMap<String, BTreeSet<String>>,
    globs: BTreeSet<String>,
}

impl Modules {
    /// The top modules of the crate in `manifest` (its folder), declared in
    /// `src/lib.rs`, or `src/main.rs` for a binary.
    fn of(manifest: &Path) -> Self {
        let source = manifest.join("src");
        let lib = ["lib.rs", "main.rs"]
            .into_iter()
            .map(|file| source.join(file))
            .find(|path| path.exists())
            .unwrap_or_else(|| panic!("{} holds no lib.rs or main.rs", source.display()));
        let root = code(&read(&lib));
        let mut modules = Self {
            lib,
            files: BTreeMap::new(),
            names: BTreeMap::new(),
            exports: BTreeMap::new(),
            globs: BTreeSet::new(),
        };
        for name in declared_modules(&root) {
            let mut files = Vec::new();
            let file = source.join(format!("{name}.rs"));
            if file.exists() {
                files.push(file);
            }
            let folder = source.join(&name);
            if folder.is_dir() {
                list(&folder, &mut files);
            }
            let mut included = Vec::new();
            let mut text = String::new();
            for file in files.iter().filter(|file| is_rust(file)) {
                let raw = read(file);
                let folder = file.parent().unwrap_or(&source);
                included.extend(includes(&raw).into_iter().map(|path| folder.join(path)));
                text.push_str(&code(&raw));
                text.push('\n');
            }
            files.extend(included);
            files.sort();
            files.dedup();
            modules.files.insert(name.clone(), files);
            modules.names.insert(name, names(&text));
        }
        for (module, items) in reexports(&root) {
            if !modules.files.contains_key(&module) {
                continue;
            }
            match items {
                None => {
                    modules.globs.insert(module);
                }
                Some(items) => {
                    for item in items {
                        modules
                            .exports
                            .entry(item)
                            .or_default()
                            .insert(module.clone());
                    }
                }
            }
        }
        modules
    }

    /// The modules `name`'s code uses ([`Set`]).
    fn uses(&self, name: &str) -> BTreeSet<String> {
        let mut used = BTreeSet::new();
        for reached in &self.names[name] {
            if reached == "*" {
                for from in self.exports.values() {
                    used.extend(from.iter().cloned());
                }
                used.extend(self.globs.iter().cloned());
            } else if self.files.contains_key(reached) {
                used.insert(reached.clone());
            } else if let Some(from) = self.exports.get(reached) {
                used.extend(from.iter().cloned());
            } else {
                used.extend(self.globs.iter().cloned());
            }
        }
        used.remove(name);
        used
    }

    /// `roots` and the modules they use, followed, but for `stop`. Panics
    /// where a root names no module.
    fn closure(&self, roots: &[&str], stop: &[&str]) -> BTreeSet<String> {
        let mut found = BTreeSet::new();
        let mut queue: Vec<String> = roots.iter().map(|root| (*root).to_owned()).collect();
        while let Some(name) = queue.pop() {
            assert!(
                self.files.contains_key(&name),
                "the crate's root declares no module {name}"
            );
            if stop.contains(&name.as_str()) || !found.insert(name.clone()) {
                continue;
            }
            queue.extend(self.uses(&name));
        }
        found
    }
}

/// Whether `path` is a Rust file.
fn is_rust(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension == "rs")
}

/// The modules `root` (a crate root's code) declares from files
/// (`mod name;`, whatever its visibility).
fn declared_modules(root: &str) -> Vec<String> {
    let tokens = tokens(root);
    let mut names = Vec::new();
    for window in tokens.windows(3) {
        if window[0] == "mod" && window[2] == ";" && is_identifier(&window[1]) {
            names.push(window[1].clone());
        }
    }
    names
}

/// The modules `root` (a crate root's code) re-exports from (`pub use
/// module::...;`), each with the items it re-exports, `None` for the
/// whole module (`*`).
fn reexports(root: &str) -> Vec<(String, Option<Vec<String>>)> {
    let tokens = tokens(root);
    let mut found = Vec::new();
    let mut at = 0;
    while at + 1 < tokens.len() {
        if tokens[at] != "use" || !is_identifier(&tokens[at + 1]) {
            at += 1;
            continue;
        }
        let end = tokens[at..]
            .iter()
            .position(|token| token == ";")
            .map_or(tokens.len(), |offset| at + offset);
        let statement = &tokens[at + 1..end];
        let module = statement[0].clone();
        if module == "crate" || module == "self" || module == "super" {
            at = end;
            continue;
        }
        if statement.iter().any(|token| token == "*") {
            found.push((module, None));
        } else {
            let items = statement[1..]
                .iter()
                .filter(|token| is_identifier(token) && token.as_str() != "as")
                .cloned()
                .collect();
            found.push((module, Some(items)));
        }
        at = end;
    }
    found
}

/// The names `text` (code without comments and strings) reaches through
/// its crate's root ([`Set`]).
fn names(text: &str) -> BTreeSet<String> {
    let tokens = tokens(text);
    let mut found = BTreeSet::new();
    let mut at = 0;
    while at + 2 < tokens.len() {
        let rooted = (tokens[at] == "crate" || tokens[at] == "super") && tokens[at + 1] == "::";
        if !rooted {
            at += 1;
            continue;
        }
        let next = &tokens[at + 2];
        if next == "{" {
            // A group's first names, at its own depth.
            let (mut depth, mut first) = (0_u32, true);
            let mut inside = at + 2;
            while let Some(token) = tokens.get(inside) {
                match token.as_str() {
                    "{" => {
                        depth += 1;
                        first = depth == 1;
                    }
                    "}" => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    "," if depth == 1 => first = true,
                    _ if depth == 1 && first && is_identifier(token) => {
                        found.insert(token.clone());
                        first = false;
                    }
                    _ => {}
                }
                inside += 1;
            }
            at = inside;
        } else {
            // A glob (`use super::*`) reaches whatever the root holds.
            if is_identifier(next) || next == "*" {
                found.insert(next.clone());
            }
            at += 2;
        }
    }
    found
}

/// `text` split into identifiers, `::` and single other characters,
/// whitespace dropped.
fn tokens(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut at = 0;
    while at < chars.len() {
        let c = chars[at];
        if c.is_whitespace() {
            at += 1;
        } else if c.is_alphanumeric() || c == '_' {
            let start = at;
            while at < chars.len() && (chars[at].is_alphanumeric() || chars[at] == '_') {
                at += 1;
            }
            tokens.push(chars[start..at].iter().collect());
        } else if c == ':' && chars.get(at + 1) == Some(&':') {
            tokens.push("::".to_owned());
            at += 2;
        } else {
            tokens.push(c.to_string());
            at += 1;
        }
    }
    tokens
}

fn is_identifier(token: &str) -> bool {
    token
        .chars()
        .next()
        .is_some_and(|first| first.is_alphabetic() || first == '_')
}

/// The files `text` (a Rust file) includes by literal path
/// (`include_str!("...")`, `include_bytes!` and `include!`), as written.
fn includes(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    for macro_name in ["include_str!(", "include_bytes!(", "include!("] {
        let mut rest = text;
        while let Some(at) = rest.find(macro_name) {
            rest = &rest[at + macro_name.len()..];
            let trimmed = rest.trim_start();
            if let Some(quoted) = trimmed.strip_prefix('"')
                && let Some((path, _)) = quoted.split_once('"')
            {
                found.push(path.to_owned());
            }
        }
    }
    found
}

/// `text` (Rust) without its comments and the contents of its string and
/// character literals, each replaced by a space.
fn code(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    while at < chars.len() {
        let c = chars[at];
        let next = chars.get(at + 1).copied();
        if c == '/' && next == Some('/') {
            while at < chars.len() && chars[at] != '\n' {
                at += 1;
            }
            out.push(' ');
        } else if c == '/' && next == Some('*') {
            at += 2;
            while at < chars.len() && !(chars[at] == '*' && chars.get(at + 1) == Some(&'/')) {
                at += 1;
            }
            at += 2;
            out.push(' ');
        } else if c == 'r' && (next == Some('"') || next == Some('#')) && raw_string(&chars, at) {
            let mut hashes = 0;
            at += 1;
            while chars.get(at) == Some(&'#') {
                hashes += 1;
                at += 1;
            }
            at += 1;
            while at < chars.len() {
                if chars[at] == '"' && (1..=hashes).all(|offset| chars.get(at + offset) == Some(&'#'))
                {
                    at += 1 + hashes;
                    break;
                }
                at += 1;
            }
            out.push(' ');
        } else if c == '"' {
            at += 1;
            while at < chars.len() && chars[at] != '"' {
                if chars[at] == '\\' {
                    at += 1;
                }
                at += 1;
            }
            at += 1;
            out.push(' ');
        } else if c == '\'' && char_literal(&chars, at) {
            at += 1;
            while at < chars.len() && chars[at] != '\'' {
                if chars[at] == '\\' {
                    at += 1;
                }
                at += 1;
            }
            at += 1;
            out.push(' ');
        } else {
            out.push(c);
            at += 1;
        }
    }
    out
}

/// Whether the `r` at `at` starts a raw string (`r"`, `r#"`), not an
/// identifier ending in `r` or starting with it.
fn raw_string(chars: &[char], at: usize) -> bool {
    let before = at.checked_sub(1).map(|before| chars[before]);
    if before.is_some_and(|before| before.is_alphanumeric() || before == '_') {
        return false;
    }
    let mut after = at + 1;
    while chars.get(after) == Some(&'#') {
        after += 1;
    }
    chars.get(after) == Some(&'"')
}

/// Whether the `'` at `at` opens a character literal rather than a
/// lifetime or a label.
fn char_literal(chars: &[char], at: usize) -> bool {
    match chars.get(at + 1) {
        Some('\\') => true,
        Some(_) => chars.get(at + 2) == Some(&'\''),
        None => false,
    }
}

/// `digest`'s bytes as a Rust array's items.
fn bytes(digest: &[u8]) -> String {
    digest
        .iter()
        .map(|byte| format!("{byte:#04x}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Writes `source` into the build's output folder as `file`.
fn write_out(file: &str, source: &str) {
    let out = PathBuf::from(var("OUT_DIR")).join(file);
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
