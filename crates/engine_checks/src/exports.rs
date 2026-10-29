use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use engine_core::{Error, ErrorCode, Expected};

use crate::{Facade, Repository};

pub fn check(repository: &Repository, facade: &Facade) -> Result<(), String> {
    let folder = |name: &str| {
        repository
            .package(name)
            .map(|package| package.folder.join("src"))
            .ok_or_else(|| format!("the facade's package {name} is not in the workspace"))
    };
    let core = core_names(&folder(facade.inner)?)?;
    let lists = facade_lists(&repository.root, &folder(facade.package)?, facade.inner)?;
    let root = repository
        .relative(&folder(facade.package)?.join("lib.rs"))
        .unwrap_or_default();
    let internal = facade.internal;
    let mut problems = Vec::new();
    if let Some(pair) = internal.windows(2).find(|pair| pair[0] >= pair[1]) {
        problems.push(
            Error::new(ErrorCode::FacadeExport, "$.facade.internal")
                .expected(Expected::Sorted)
                .got(pair[1]),
        );
    }
    for name in internal {
        if !core.contains(*name) {
            problems.push(Error::new(ErrorCode::ExceptionUnused, "$.facade.internal").got(name));
        }
    }
    for name in &core {
        let places = lists.get(name).map_or(&[][..], Vec::as_slice);
        match (internal.contains(&name.as_str()), places) {
            (true, []) | (false, [_]) => {}
            (true, places) => problems.push(
                Error::new(ErrorCode::FacadeExport, places[0].as_str())
                    .expected(Expected::Absent)
                    .got(name),
            ),
            (false, []) => problems.push(
                Error::new(ErrorCode::FacadeExport, root.as_str())
                    .expected(Expected::Present)
                    .got(name),
            ),
            (false, places) => problems.push(
                Error::new(ErrorCode::FacadeExport, places[1].as_str())
                    .expected(Expected::Unique)
                    .got(name),
            ),
        }
    }
    if problems.is_empty() {
        let record = serde_json::json!({
            "event": "facade_exports_once",
            "inner": facade.inner,
            "public": core.len(),
            "internal": internal.len(),
        });
        println!("{record}");
        Ok(())
    } else {
        crate::report(&problems)
    }
}

fn code(path: &Path) -> Result<String, String> {
    let source = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    Ok(crate::policy::rust::split(&source).code.join("\n"))
}

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

enum Declared {
    Name(String),
    Glob(String),
}

fn core_names(source: &Path) -> Result<BTreeSet<String>, String> {
    let mut names = BTreeSet::new();
    module_names(&source.join("lib.rs"), source, &mut names)?;
    for file in rust_files(source)? {
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

fn child_module(children: &Path, name: &str) -> Option<(PathBuf, PathBuf)> {
    let file = children.join(format!("{name}.rs"));
    if file.is_file() {
        return Some((file, children.join(name)));
    }
    let file = children.join(name).join("mod.rs");
    file.is_file().then(|| (file, children.join(name)))
}

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

fn definition_before(before: &str) -> bool {
    let name_start = before
        .rfind(|c: char| !(c.is_alphanumeric() || c == '_'))
        .map_or(0, |at| at + 1);
    before[..name_start].trim_end().ends_with("macro_rules!")
}

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

fn identifier(text: &str) -> &str {
    let end = text
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(text.len());
    &text[..end]
}

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

fn use_names(tree: &str) -> Vec<String> {
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

fn facade_lists(
    root: &Path,
    source: &Path,
    inner: &str,
) -> Result<BTreeMap<String, Vec<String>>, String> {
    let mut lists: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let marker = format!("pub use {inner}::");
    for file in rust_files(source)? {
        let code = code(&file)?;
        let shown = file
            .strip_prefix(root)
            .unwrap_or(&file)
            .display()
            .to_string();
        for (at, _) in code.match_indices(marker.as_str()) {
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
