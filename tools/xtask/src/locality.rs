//! The Stress Lab's content locality: a world module (each module the world
//! registry, `apps/stress-lab/src/world.rs`, names) imports only the content
//! kit (`crate::kit`), the engine's crates, the standard library and
//! itself, so no world reaches into another world's code or the workload's.
//! What worlds share goes in the kit.

use std::fs;
use std::path::{Path, PathBuf};

/// The app whose worlds are checked, and its registry and kit.
const SOURCE: &str = "apps/stress-lab/src";
const REGISTRY: &str = "apps/stress-lab/src/world.rs";
const MANIFEST: &str = "apps/stress-lab/Cargo.toml";
const KIT: &str = "kit";

/// Checks every world module's files.
pub fn check(root: &Path) -> Result<(), String> {
    let worlds = worlds(root)?;
    let foreign = foreign_crates(root)?;
    let mut problems = Vec::new();
    let mut files = 0;
    for world in &worlds {
        for (path, depth) in world_files(&root.join(SOURCE), world)? {
            files += 1;
            let text = fs::read_to_string(&path)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            let shown = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .display()
                .to_string()
                .replace('\\', "/");
            for (line, what) in imports(&code(&text), world, depth, &foreign) {
                problems.push(format!("{shown}:{line}: {what}"));
            }
        }
    }
    if problems.is_empty() {
        println!(
            "The {} worlds' {files} files import only the content kit, the engine and themselves.",
            worlds.len()
        );
        return Ok(());
    }
    Err(format!(
        "{}\na world module imports only the content kit ({SOURCE}/kit/), the engine and itself: move what worlds share into the kit",
        problems.join("\n")
    ))
}

/// The world modules the registry names: `Self::Name => &module::Type`.
fn worlds(root: &Path) -> Result<Vec<String>, String> {
    let text = fs::read_to_string(root.join(REGISTRY))
        .map_err(|error| format!("cannot read {REGISTRY}: {error}"))?;
    let worlds: Vec<String> = text
        .lines()
        .filter_map(|line| {
            let (_, target) = line.trim().strip_prefix("Self::")?.split_once("=> &")?;
            let (module, _) = target.split_once("::")?;
            Some(module.trim().to_owned())
        })
        .collect();
    if worlds.is_empty() {
        return Err(format!(
            "{REGISTRY} names no world (`Self::Name => &module::Type,` arms)"
        ));
    }
    Ok(worlds)
}

/// The app's dependencies other than the engine's crates, as paths name
/// them: what no world may use.
fn foreign_crates(root: &Path) -> Result<Vec<String>, String> {
    let text = fs::read_to_string(root.join(MANIFEST))
        .map_err(|error| format!("cannot read {MANIFEST}: {error}"))?;
    let mut in_dependencies = false;
    let mut found = Vec::new();
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') {
            in_dependencies = line.ends_with("dependencies]");
            continue;
        }
        let Some((name, _)) = line.split_once('=') else {
            continue;
        };
        let name = name.trim().split('.').next().unwrap_or_default();
        if in_dependencies && !name.is_empty() && !name.starts_with('#') {
            let name = name.replace('-', "_");
            if name != "engine" && !name.starts_with("engine_") {
                found.push(name);
            }
        }
    }
    Ok(found)
}

/// A world's source files and each one's module depth below the world's
/// own module: `world.rs` or `world/mod.rs` 0, `world/part.rs` 1...
fn world_files(source: &Path, world: &str) -> Result<Vec<(PathBuf, usize)>, String> {
    let mut found = Vec::new();
    let file = source.join(format!("{world}.rs"));
    if file.is_file() {
        found.push((file, 0));
    }
    let folder = source.join(world);
    if folder.is_dir() {
        collect(&folder, 0, &mut found)?;
    }
    if found.is_empty() {
        return Err(format!(
            "{REGISTRY} names world module {world}, which has no {world}.rs or {world}/"
        ));
    }
    Ok(found)
}

/// The Rust files under `folder`, `depth` modules below the world's.
fn collect(folder: &Path, depth: usize, found: &mut Vec<(PathBuf, usize)>) -> Result<(), String> {
    let mut entries: Vec<PathBuf> = fs::read_dir(folder)
        .map_err(|error| format!("cannot read {}: {error}", folder.display()))?
        .flatten()
        .map(|entry| entry.path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect(&path, depth + 1, found)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let module = path.file_stem().is_some_and(|stem| stem != "mod");
            found.push((path, depth + usize::from(module)));
        }
    }
    Ok(())
}

/// What the code `text` of a module `depth` below world `world` imports
/// that it may not: each by its line.
fn imports(text: &str, world: &str, depth: usize, foreign: &[String]) -> Vec<(usize, String)> {
    let mut found = Vec::new();
    let line_of = |at: usize| text[..at].matches('\n').count() + 1;
    for (at, _) in text.match_indices("crate::") {
        if !boundary(text, at) {
            continue;
        }
        for module in roots(&text[at + "crate::".len()..]) {
            if module != KIT && module != world {
                found.push((line_of(at), format!("crate::{module}")));
            }
        }
    }
    for (at, _) in text.match_indices("super::") {
        if !boundary(text, at) || text[..at].ends_with("super::") {
            continue;
        }
        let climbs = text.as_bytes()[at..]
            .chunks(7)
            .take_while(|chunk| *chunk == b"super::")
            .count();
        if climbs > depth {
            found.push((
                line_of(at),
                format!(
                    "{} (above the world's own module)",
                    "super::".repeat(climbs)
                ),
            ));
        }
    }
    for name in foreign {
        let path = format!("{name}::");
        for (at, _) in text.match_indices(&path) {
            if boundary(text, at) {
                found.push((line_of(at), format!("{name} (not an engine crate)")));
            }
        }
    }
    for marker in ["extern crate", "#[path"] {
        for (at, _) in text.match_indices(marker) {
            found.push((line_of(at), marker.to_owned()));
        }
    }
    found.sort();
    found
}

/// Whether the path starting at `at` begins a path: no identifier character
/// or `::` right before it.
fn boundary(text: &str, at: usize) -> bool {
    let before = &text[..at];
    !before.ends_with("::")
        && !before
            .chars()
            .next_back()
            .is_some_and(|character| character.is_alphanumeric() || character == '_')
}

/// The first segments a path after `crate::` names: one, or each item of a
/// `{...}` group.
fn roots(path: &str) -> Vec<String> {
    let path = path.trim_start();
    let Some(group) = path.strip_prefix('{') else {
        let name: String = path
            .chars()
            .take_while(|character| character.is_alphanumeric() || *character == '_')
            .collect();
        return vec![name];
    };
    let mut items = Vec::new();
    let (mut depth, mut start) = (0usize, 0usize);
    for (index, character) in group.char_indices() {
        match character {
            '{' => depth += 1,
            '}' if depth == 0 => {
                items.push(&group[start..index]);
                break;
            }
            '}' => depth -= 1,
            ',' if depth == 0 => {
                items.push(&group[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    items
        .into_iter()
        .map(|item| {
            item.trim()
                .chars()
                .take_while(|character| character.is_alphanumeric() || *character == '_')
                .collect::<String>()
        })
        .filter(|name| !name.is_empty())
        .collect()
}

/// `text` with its comments, strings and character literals blanked
/// (newlines kept), so paths inside them do not count.
fn code(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = bytes.to_vec();
    let blank = |out: &mut Vec<u8>, from: usize, to: usize| {
        for byte in &mut out[from..to.min(bytes.len())] {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
    };
    let mut at = 0;
    while at < bytes.len() {
        let rest = &bytes[at..];
        if rest.starts_with(b"//") {
            let end = rest
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |offset| at + offset);
            blank(&mut out, at, end);
            at = end;
        } else if rest.starts_with(b"/*") {
            let (mut depth, mut end) = (0usize, at);
            while end < bytes.len() {
                if bytes[end..].starts_with(b"/*") {
                    depth += 1;
                    end += 2;
                } else if bytes[end..].starts_with(b"*/") {
                    depth -= 1;
                    end += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    end += 1;
                }
            }
            blank(&mut out, at, end);
            at = end;
        } else if rest.starts_with(b"r#") || rest.starts_with(b"r\"") {
            let hashes = rest[1..].iter().take_while(|byte| **byte == b'#').count();
            if rest.get(1 + hashes) == Some(&b'"') {
                let close: Vec<u8> = std::iter::once(b'"')
                    .chain(std::iter::repeat_n(b'#', hashes))
                    .collect();
                let body = at + 2 + hashes;
                let end = bytes[body..]
                    .windows(close.len())
                    .position(|window| window == close.as_slice())
                    .map_or(bytes.len(), |offset| body + offset + close.len());
                blank(&mut out, at, end);
                at = end;
            } else {
                at += 1;
            }
        } else if rest[0] == b'"' {
            let mut end = at + 1;
            while end < bytes.len() && bytes[end] != b'"' {
                end += if bytes[end] == b'\\' { 2 } else { 1 };
            }
            blank(&mut out, at, end + 1);
            at = end + 1;
        } else if rest[0] == b'\'' {
            // A character literal ('x', '\n', '\u{..}'); a lifetime has no
            // closing quote.
            let end = if rest.get(1) == Some(&b'\\') {
                rest[2..]
                    .iter()
                    .position(|byte| *byte == b'\'')
                    .map(|offset| at + 2 + offset + 1)
            } else {
                let width = std::str::from_utf8(&rest[1..rest.len().min(5)])
                    .ok()
                    .or_else(|| std::str::from_utf8(&rest[1..rest.len().min(2)]).ok())
                    .and_then(|tail| tail.chars().next())
                    .map_or(1, char::len_utf8);
                (rest.get(1 + width) == Some(&b'\'')).then_some(at + 2 + width)
            };
            match end {
                Some(end) => {
                    blank(&mut out, at, end);
                    at = end;
                }
                None => at += 1,
            }
        } else {
            at += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}
