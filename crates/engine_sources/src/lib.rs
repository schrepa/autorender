#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

pub fn record_key(source: &[u8; 32], kind: &str, identity: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(source);
    hash.update(kind.as_bytes());
    hash.update([0]);
    hash.update(identity);
    hash.finalize().into()
}

#[derive(Clone, Copy, Debug)]
pub struct Set<'a> {
    pub name: &'a str,
    pub roots: &'a [&'a str],
    pub stop: &'a [&'a str],
    pub files: &'a [&'a str],
    pub crates: Option<&'a [&'a str]>,
}

pub fn emit() {
    emit_sets(&[]);
}

pub fn emit_sets(sets: &[Set<'_>]) {
    let manifest = PathBuf::from(var("CARGO_MANIFEST_DIR"));
    let mut source = source_declaration(&manifest, "this crate");
    if !sets.is_empty() {
        let modules = Modules::of(&manifest);
        source.push_str(
            "/// The hashes of the code making each kind of this crate's records:\n\
             /// the modules making it and what they use (`engine_sources::Set`).\n\
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

pub fn crate_source(folder: &Path) -> [u8; 32] {
    let root = workspace_root(folder);
    let mut files = Vec::new();
    for folder in &crate_folders(&root, folder, None) {
        list(folder, &mut files);
    }
    hash_files(&root, folder, files)
}

fn digest(root: &Path, manifest: &Path, files: Vec<PathBuf>) -> [u8; 32] {
    println!(
        "cargo:rerun-if-changed={}",
        root.join("rust-toolchain.toml").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("Cargo.lock").display()
    );
    hash_files(root, manifest, files)
}

fn hash_files(root: &Path, manifest: &Path, files: Vec<PathBuf>) -> [u8; 32] {
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
    named.insert(
        "rust-toolchain.toml".to_owned(),
        root.join("rust-toolchain.toml"),
    );
    let mut hash = Sha256::new();
    for (name, path) in &named {
        let bytes = fs::read(path).unwrap_or_else(|error| panic!("cannot read {name}: {error}"));
        hash.update(name.as_bytes());
        hash.update([0]);
        hash.update(normalized(&bytes));
        hash.update([0]);
    }
    let lock = root.join("Cargo.lock");
    let package = package_name(&read(&manifest.join("Cargo.toml")));
    hash.update(b"Cargo.lock");
    hash.update([0]);
    hash.update(lock_entries(&read(&lock), &package).as_bytes());
    hash.finalize().into()
}

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

struct Modules {
    lib: PathBuf,
    files: BTreeMap<String, Vec<PathBuf>>,
    names: BTreeMap<String, BTreeSet<String>>,
    exports: BTreeMap<String, BTreeSet<String>>,
    globs: BTreeSet<String>,
}

impl Modules {
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
            if test_code(&name) {
                continue;
            }
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

fn is_rust(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension == "rs")
}

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
            if is_identifier(next) || next == "*" {
                found.insert(next.clone());
            }
            at += 2;
        }
    }
    found
}

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
                if chars[at] == '"'
                    && (1..=hashes).all(|offset| chars.get(at + offset) == Some(&'#'))
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

fn char_literal(chars: &[char], at: usize) -> bool {
    match chars.get(at + 1) {
        Some('\\') => true,
        Some(_) => chars.get(at + 2) == Some(&'\''),
        None => false,
    }
}

fn bytes(digest: &[u8]) -> String {
    digest
        .iter()
        .map(|byte| format!("{byte:#04x}"))
        .collect::<Vec<_>>()
        .join(", ")
}

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

fn quoted_after(value: &str, key: &str) -> Option<String> {
    let at = value.find(key)?;
    let rest = value[at + key.len()..].trim_start().strip_prefix('=')?;
    let rest = rest.trim_start().strip_prefix('"')?;
    rest.split_once('"').map(|(path, _)| path.to_owned())
}

fn test_code(name: &str) -> bool {
    let stem = name.strip_suffix(".rs").unwrap_or(name);
    ["smoke", "smokes", "tests"].contains(&stem)
        || stem.ends_with("_smoke")
        || stem.ends_with("_smokes")
}

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
        if scratch || test_code(&name) {
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

fn normalized(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    for (index, &byte) in bytes.iter().enumerate() {
        if !(byte == b'\r' && bytes.get(index + 1) == Some(&b'\n')) {
            out.push(byte);
        }
    }
    out
}
