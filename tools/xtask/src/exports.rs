//! The facade exports engine_core's public names once each: every name
//! engine_core's root exports is in exactly one of the facade's lists (a
//! `pub use engine_core::…;` in crates/engine/src: the games' list the root
//! and prelude share, the lower-level list, terrain's, render's...) or in
//! [`INTERNAL`]. A name added to engine_core fails lint until it is given
//! an audience.
//!
//! engine_core's names are read from its sources as text: the root's public
//! items and re-exports, those of each module a `pub use module::*`
//! exports (the items its macro calls and macros declare included), and
//! every exported macro.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// engine_core's public names the facade leaves out: what engine crates,
/// hosts and tools use. Sorted.
const INTERNAL: &[&str] = &[
    "CHUNK_LENGTH",
    "DevStepError",
    "GroundedAt",
    "MAX_DEV_TICKS",
    "atmosphere_fields",
    "engine_features",
    "install",
    "state_atmosphere_rules",
];

/// Fails at a public name of engine_core that no facade list or
/// [`INTERNAL`] holds, or that more than one does.
pub fn check(root: &Path) -> Result<(), String> {
    let core = core_names(root)?;
    let lists = facade_lists(root)?;
    let mut problems = Vec::new();
    if INTERNAL.windows(2).any(|pair| pair[0] >= pair[1]) {
        problems.push("INTERNAL (tools/xtask/src/exports.rs) must be sorted and unique".to_owned());
    }
    for name in INTERNAL {
        if !core.contains(*name) {
            problems.push(format!(
                "INTERNAL lists {name}, which engine_core does not export"
            ));
        }
    }
    for name in &core {
        let places = lists.get(name).map_or(&[][..], Vec::as_slice);
        match (INTERNAL.contains(&name.as_str()), places) {
            (true, []) | (false, [_]) => {}
            (true, places) => problems.push(format!(
                "{name} is in INTERNAL and exported by the facade ({})",
                places.join(", ")
            )),
            (false, []) => problems.push(format!(
                "engine_core exports {name}, which no facade list does: add it to the games' or \
                 the lower-level list (crates/engine/src/lib.rs), a module's list, or INTERNAL \
                 (tools/xtask/src/exports.rs)"
            )),
            (false, places) => problems.push(format!(
                "{name} is in {} facade lists ({}): keep it in one",
                places.len(),
                places.join(", ")
            )),
        }
    }
    if problems.is_empty() {
        println!(
            "The facade exports each of engine_core's {} public names once, but the {} internal ones.",
            core.len(),
            INTERNAL.len()
        );
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// A Rust file's code without comments, literals emptied.
fn code(path: &Path) -> Result<String, String> {
    let source = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    Ok(policy::rust::split(&source).code.join("\n"))
}

/// The Rust files under `directory`.
fn rust_files(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let mut pending = vec![directory.to_path_buf()];
    while let Some(path) = pending.pop() {
        if path.is_dir() {
            let entries = fs::read_dir(&path)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            for entry in entries {
                pending.push(entry.map_err(|error| error.to_string())?.path());
            }
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

/// What a module declares public: a name, or every public name of a module
/// it re-exports whole (`pub use child::*`).
enum Declared {
    Name(String),
    Glob(String),
}

/// Every public name engine_core's root exports.
fn core_names(root: &Path) -> Result<BTreeSet<String>, String> {
    let source = root.join("crates/engine_core/src");
    let mut names = BTreeSet::new();
    module_names(&source.join("lib.rs"), &source, &mut names)?;
    // Exported macros are the root's wherever they are declared.
    for file in rust_files(&source)? {
        let code = code(&file)?;
        for (at, _) in code.match_indices("#[macro_export]") {
            if let Some(rest) = code[at + "#[macro_export]".len()..]
                .trim_start()
                .strip_prefix("macro_rules!")
            {
                names.insert(identifier(rest.trim_start()).to_owned());
            }
        }
    }
    Ok(names)
}

/// The public names of the module in `file`, whose child modules are in
/// `children`, its globs followed.
fn module_names(file: &Path, children: &Path, names: &mut BTreeSet<String>) -> Result<(), String> {
    let mut declared = Vec::new();
    scan(&code(file)?, &mut declared);
    for item in declared {
        match item {
            Declared::Name(name) => {
                names.insert(name);
            }
            Declared::Glob(module) => {
                let (file, children) = child_module(children, &module).ok_or_else(|| {
                    format!(
                        "{}: cannot find module {module} of a `pub use {module}::*`",
                        file.display()
                    )
                })?;
                module_names(&file, &children, names)?;
            }
        }
    }
    Ok(())
}

/// The file of module `name`, declared in a file whose child modules are in
/// `children`, and the folder of its own children.
fn child_module(children: &Path, name: &str) -> Option<(PathBuf, PathBuf)> {
    let file = children.join(format!("{name}.rs"));
    if file.is_file() {
        return Some((file, children.join(name)));
    }
    let file = children.join(name).join("mod.rs");
    file.is_file().then(|| (file, children.join(name)))
}

/// The public declarations at brace depth 0 of `code`: `pub` items, `pub
/// use` trees, and those of the bodies of macro calls (`settings! { pub
/// struct ... }`) and of macro definitions' expansions.
fn scan(code: &str, declared: &mut Vec<Declared>) {
    let mut at = 0;
    while at < code.len() {
        let rest = &code[at..];
        let word_start = rest.starts_with(|c: char| c.is_alphabetic() || c == '_')
            && !code[..at]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
        if word_start && identifier(rest) == "pub" && rest[3..].starts_with(char::is_whitespace) {
            let declaration = rest[3..].trim_start();
            if let Some(tree) = declaration.strip_prefix("use")
                && tree.starts_with(char::is_whitespace)
            {
                let end = tree.find(';').unwrap_or(tree.len());
                for name in use_names(&tree[..end]) {
                    declared.push(match name.strip_suffix("::*") {
                        Some(module) => Declared::Glob(module.to_owned()),
                        None => Declared::Name(name),
                    });
                }
            } else if let Some(name) = item_name(declaration) {
                declared.push(Declared::Name(name.to_owned()));
            }
            at += 3;
            continue;
        }
        if rest.starts_with('{') {
            let end = at + matching_brace(rest);
            let before = code[..at].trim_end();
            let body = &code[at + 1..end.saturating_sub(1).max(at + 1)];
            if before.ends_with('!') {
                scan(body, declared);
            } else if definition_before(before) {
                // A macro definition: each arm's expansion.
                for expansion in expansions(body) {
                    scan(expansion, declared);
                }
            }
            at = end;
            continue;
        }
        at += rest.chars().next().map_or(1, char::len_utf8);
    }
}

/// Whether `before` ends with `macro_rules! name`.
fn definition_before(before: &str) -> bool {
    let name_start = before
        .rfind(|c: char| !(c.is_alphanumeric() || c == '_'))
        .map_or(0, |at| at + 1);
    before[..name_start].trim_end().ends_with("macro_rules!")
}

/// The expansions of a macro definition's arms: the block after each `=>`.
fn expansions(body: &str) -> Vec<&str> {
    let mut expansions = Vec::new();
    let mut at = 0;
    while let Some(arrow) = body[at..].find("=>") {
        let start = at + arrow + 2;
        let Some(open) = body[start..].find('{') else {
            break;
        };
        let open = start + open;
        let end = open + matching_brace(&body[open..]);
        expansions.push(&body[open + 1..end.saturating_sub(1).max(open + 1)]);
        at = end;
    }
    expansions
}

/// The length of the block `text` opens with, through its closing brace.
fn matching_brace(text: &str) -> usize {
    let mut depth = 0usize;
    for (at, c) in text.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return at + 1;
                }
            }
            _ => {}
        }
    }
    text.len()
}

/// The leading identifier of `text`.
fn identifier(text: &str) -> &str {
    let end = text
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(text.len());
    &text[..end]
}

/// The name a public item's declaration (after `pub`) declares.
fn item_name(declaration: &str) -> Option<&str> {
    let mut rest = declaration;
    loop {
        let word = identifier(rest);
        let after = rest[word.len()..].trim_start();
        match word {
            "struct" | "enum" | "trait" | "type" | "fn" | "static" | "mod" | "union" => {
                let name = identifier(after);
                return (!name.is_empty()).then_some(name);
            }
            // `const fn`, or a constant.
            "const" if identifier(after) == "fn" => rest = after,
            "const" => {
                let name = identifier(after);
                return (!name.is_empty()).then_some(name);
            }
            "async" | "unsafe" => rest = after,
            "extern" => rest = after.trim_start_matches("\"\"").trim_start(),
            _ => return None,
        }
    }
}

/// The names a use tree brings in: each path's last segment or its alias,
/// `self` naming the path before it, and globs as `path::*`.
fn use_names(tree: &str) -> Vec<String> {
    // `x as y` becomes `x@y`, then whitespace goes.
    let normalized: String = tree
        .split_whitespace()
        .map(|word| if word == "as" { "@" } else { word })
        .collect();
    let mut names = Vec::new();
    walk_tree("", &normalized, &mut names);
    names
}

fn walk_tree(prefix: &str, tree: &str, names: &mut Vec<String>) {
    if let Some(open) = tree.find('{') {
        let prefix = join(prefix, tree[..open].trim_end_matches("::"));
        let inner = tree[open + 1..]
            .strip_suffix('}')
            .unwrap_or(&tree[open + 1..]);
        let (mut depth, mut start) = (0usize, 0);
        for (at, c) in inner.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => depth = depth.saturating_sub(1),
                ',' if depth == 0 => {
                    walk_tree(&prefix, &inner[start..at], names);
                    start = at + 1;
                }
                _ => {}
            }
        }
        walk_tree(&prefix, &inner[start..], names);
        return;
    }
    if tree.is_empty() {
        return;
    }
    let path = join(prefix, tree);
    if let Some(module) = path.strip_suffix("::*") {
        names.push(format!("{module}::*"));
        return;
    }
    let last = path.rsplit("::").next().unwrap_or(&path);
    if let Some((_, alias)) = last.split_once('@') {
        names.push(alias.to_owned());
    } else if last == "self" {
        let module = path.trim_end_matches("::self");
        names.push(module.rsplit("::").next().unwrap_or(module).to_owned());
    } else {
        names.push(last.to_owned());
    }
}

fn join(prefix: &str, path: &str) -> String {
    match (prefix.is_empty(), path.is_empty()) {
        (true, _) => path.to_owned(),
        (false, true) => prefix.to_owned(),
        (false, false) => format!("{prefix}::{path}"),
    }
}

/// Each engine_core root name the facade exports, with where: each `pub
/// use engine_core::...;` in crates/engine/src is one list.
fn facade_lists(root: &Path) -> Result<BTreeMap<String, Vec<String>>, String> {
    let mut lists: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for file in rust_files(&root.join("crates/engine/src"))? {
        let code = code(&file)?;
        let shown = file
            .strip_prefix(root)
            .unwrap_or(&file)
            .display()
            .to_string();
        let marker = "pub use engine_core::";
        for (at, _) in code.match_indices(marker) {
            let line = code[..at].matches('\n').count() + 1;
            let tree = &code[at + marker.len()..];
            let tree = &tree[..tree.find(';').unwrap_or(tree.len())];
            for name in root_names(tree) {
                lists
                    .entry(name)
                    .or_default()
                    .push(format!("{}:{line}", shown.replace('\\', "/")));
            }
        }
    }
    Ok(lists)
}

/// The engine_core root names `pub use engine_core::TREE;` exports: the
/// tree's own entries (`{A, B}`, `A`, `A as B`), not the names inside a
/// root name (`glam::{Vec3}` exports glam's, not glam).
fn root_names(tree: &str) -> Vec<String> {
    let tree = tree.trim();
    let entries: Vec<&str> = match tree
        .strip_prefix('{')
        .and_then(|inner| inner.strip_suffix('}'))
    {
        Some(inner) => {
            let (mut entries, mut depth, mut start) = (Vec::new(), 0usize, 0);
            for (at, c) in inner.char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => depth = depth.saturating_sub(1),
                    ',' if depth == 0 => {
                        entries.push(&inner[start..at]);
                        start = at + 1;
                    }
                    _ => {}
                }
            }
            entries.push(&inner[start..]);
            entries
        }
        None => vec![tree],
    };
    entries
        .into_iter()
        .map(str::trim)
        .filter(|entry| !entry.is_empty() && !entry.contains("::"))
        .map(|entry| identifier(entry).to_owned())
        .collect()
}
