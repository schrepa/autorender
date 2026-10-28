use crate::{Problems, rust};

pub const LOCKFILES: &[&str] = &["Cargo.lock", "web/package-lock.json"];

const MARKERS: [&str; 4] = [
    concat!("TO", "DO"),
    concat!("FIX", "ME"),
    concat!("XX", "X"),
    concat!("HA", "CK"),
];

const GROUPS: [&str; 2] = ["warnings", "clippy::all"];

const ENGINE_DENIES: [&str; 2] = [
    "clippy::wildcard_enum_match_arm",
    "clippy::match_wildcard_for_single_variants",
];

const TILE_MAPS: (&str, &str) = (
    "crates/engine_core/src/features/terrain/",
    "crates/engine_core/src/features/terrain/tile_map.rs",
);

pub const UNSAFE_FILES: &[(&str, &str)] = &[
    (
        "crates/engine_profile/src/allocator.rs",
        "implements the standard allocator's unsafe contract",
    ),
    (
        "crates/engine_profiler/src/counters.rs",
        "Vulkan's layer entry points and calls",
    ),
    (
        "crates/engine_profiler/src/load.rs",
        "Windows' process, processor time and GPU counter calls",
    ),
    (
        "crates/engine_profiler/src/pix.rs",
        "loads PIX's capturer and calls its capture functions",
    ),
    (
        "crates/engine_profiler/src/renderdoc.rs",
        "calls RenderDoc's capture functions",
    ),
    (
        "crates/engine_profiler/src/sampler/linux.rs",
        "signals threads and reads their registers",
    ),
    (
        "crates/engine_profiler/src/sampler/macos.rs",
        "suspends threads and reads their registers",
    ),
    (
        "crates/engine_profiler/src/sampler/mod.rs",
        "follows frame pointers within a thread's stack and reads sampled code",
    ),
    (
        "crates/engine_profiler/src/sampler/windows.rs",
        "suspends threads, reads their registers and unwinds them",
    ),
    (
        "crates/engine_profiler/src/stack.rs",
        "queries the calling thread's stack limits and committed pages",
    ),
    (
        "crates/engine_render/src/renderer/native.rs",
        "starts and stops a graphics debugger's capture",
    ),
];

pub fn check_text(path: &str, text: &str, problems: &mut Problems) {
    for (index, line) in text.lines().enumerate() {
        for marker in MARKERS {
            if line.match_indices(marker).any(|(at, _)| {
                let before = line[..at].chars().next_back();
                let after = line[at + marker.len()..].chars().next();
                !before.is_some_and(|c| c.is_alphanumeric() || c == '_')
                    && !after
                        .is_some_and(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
            }) {
                problems.at(
                    path,
                    index + 1,
                    &format!(
                        "a {marker} marker: do the work, or say what the code does now (work \
                         left for later belongs in an issue)"
                    ),
                );
            }
        }
    }
}

pub fn check(path: &str, source: &str, lints: &[String], problems: &mut Problems) {
    let split = rust::split(source);
    if crate_root(path) {
        let code: String = split.code.concat().split_whitespace().collect();
        for attribute in ["#![allow(", "#![expect("] {
            for (at, _) in code.match_indices(attribute) {
                let listed = &code[at + attribute.len()..];
                let listed = &listed[..listed.find(')').unwrap_or(listed.len())];
                for lint in listed.split(',').filter(|item| !item.contains('=')) {
                    if lints.iter().any(|key| covers(key, lint)) || GROUPS.contains(&lint) {
                        problems.file(
                            path,
                            &format!(
                                "allows the workspace lint {lint} crate-wide; allow it where \
                                 it is needed, with a reason"
                            ),
                        );
                    }
                }
            }
        }
        if engine_library(path) {
            let denied: Vec<&str> = code
                .match_indices("#![deny(")
                .flat_map(|(at, attribute)| {
                    let listed = &code[at + attribute.len()..];
                    listed[..listed.find(')').unwrap_or(listed.len())].split(',')
                })
                .collect();
            for lint in ENGINE_DENIES.iter().filter(|lint| !denied.contains(lint)) {
                problems.file(
                    path,
                    &format!(
                        "an engine crate denies {lint} at its root: #![deny({})]",
                        ENGINE_DENIES.join(", ")
                    ),
                );
            }
        }
    }
    if path.starts_with(TILE_MAPS.0) && path != TILE_MAPS.1 {
        let code: String = split.code.concat().split_whitespace().collect();
        if code.contains("HashMap<TileKey") {
            problems.file(
                path,
                "keys a hash map by tile: use TileMap (tile_map.rs), whose tiles walk in key order",
            );
        }
    }
    let listed = UNSAFE_FILES.iter().any(|(file, _)| *file == path);
    let first = split
        .code
        .iter()
        .position(|code| rust::has_word(code, "unsafe") || rust::has_word(code, "unsafe_code"));
    if !listed && let Some(line) = first {
        problems.at(
            path,
            line + 1,
            "unsafe code outside the files tools/policy/src/sources.rs lists (UNSAFE_FILES); \
             keep it in a listed file, or list this one with its reason",
        );
    }
    if listed && !split.code.iter().any(|code| rust::has_word(code, "unsafe")) {
        problems.at(
            path,
            1,
            "listed in UNSAFE_FILES (tools/policy/src/sources.rs) but has no unsafe code: \
             remove it from the list",
        );
    }
    for (index, code) in split.code.iter().enumerate() {
        for end in rust::words(code, "unsafe") {
            let kind = match split.token_after(index, end).as_str() {
                "{" => "block",
                "impl" => "impl",
                _ => continue,
            };
            if !split.safety_comment_above(index)
                && !split.safety_comment_above(split.statement_start(index))
            {
                problems.at(
                    path,
                    index + 1,
                    &format!(
                        "unsafe {kind} without a `// SAFETY:` comment above it saying why it is \
                         sound"
                    ),
                );
            }
        }
    }
}

const ENGINE_ERROR: &str = "crates/engine/src/error.rs";

pub struct ErrorType {
    path: String,
    line: usize,
    name: String,
}

pub fn error_types(path: &str, source: &str, found: &mut Vec<ErrorType>) {
    if !path.starts_with("crates/") || path == ENGINE_ERROR {
        return;
    }
    for (index, code) in rust::split(source).code.iter().enumerate() {
        let code = code.trim_start();
        let Some(declared) = code
            .strip_prefix("pub struct ")
            .or_else(|| code.strip_prefix("pub enum "))
        else {
            continue;
        };
        let name: String = declared
            .chars()
            .take_while(|character| character.is_alphanumeric() || *character == '_')
            .collect();
        if name.ends_with("Error") {
            found.push(ErrorType {
                path: path.to_owned(),
                line: index + 1,
                name,
            });
        }
    }
}

pub fn conversions(
    root: &std::path::Path,
    found: &[ErrorType],
    problems: &mut Problems,
) -> Result<(), String> {
    let source = std::fs::read_to_string(root.join(ENGINE_ERROR))
        .map_err(|error| format!("cannot read {ENGINE_ERROR}: {error}"))?;
    let impls: Vec<String> = rust::split(&source)
        .code
        .iter()
        .map(|code| code.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|code| code.starts_with("impl From<") && code.ends_with("> for Error {"))
        .collect();
    for error in found {
        let converted = impls.iter().any(|code| {
            let from = &code["impl From<".len()..code.len() - "> for Error {".len()];
            from == error.name || from.ends_with(&format!("::{}", error.name))
        });
        if !converted {
            problems.at(
                &error.path,
                error.line,
                &format!(
                    "{} is a public engine error without a conversion into engine::Error: \
                     add `impl From<{}> for Error` to {ENGINE_ERROR}, an exhaustive match \
                     choosing each variant's code",
                    error.name, error.name
                ),
            );
        }
    }
    Ok(())
}

fn covers(key: &str, lint: &str) -> bool {
    key == lint
        || key
            .strip_suffix("all")
            .is_some_and(|tool| tool.ends_with("::") && lint.starts_with(tool))
}

fn engine_library(path: &str) -> bool {
    path.strip_prefix("crates/")
        .and_then(|rest| rest.strip_suffix("/src/lib.rs"))
        .is_some_and(|name| !name.contains('/'))
}

fn crate_root(path: &str) -> bool {
    let (folder, name) = path.rsplit_once('/').unwrap_or(("", path));
    matches!(name, "lib.rs" | "main.rs" | "build.rs")
        || folder.ends_with("/bin")
        || folder.ends_with("/examples")
}
