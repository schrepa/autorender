//! Source rules: unsafe code stays in the files the list names and every
//! unsafe block and impl says why it is sound (in files Clippy does not
//! build on this platform too), crate roots allow no workspace lint
//! crate-wide, and no file leaves work for later behind a marker.

use crate::{Problems, rust};

/// Files the source rules skip: what package managers write.
pub const LOCKFILES: &[&str] = &["Cargo.lock", "web/package-lock.json"];

/// Markers of work left for later, which belongs in an issue or a commit,
/// not in the tree (spelled in parts so this file holds none).
const MARKERS: [&str; 4] = [
    concat!("TO", "DO"),
    concat!("FIX", "ME"),
    concat!("XX", "X"),
    concat!("HA", "CK"),
];

/// Lint groups that take every workspace lint with them.
const GROUPS: [&str; 2] = ["warnings", "clippy::all"];

/// The lints every engine crate denies at its root: a match names the
/// variants each arm takes, so a new variant fails to compile wherever it
/// needs a decision. (Apps and tools match their content tables with
/// wildcards.)
const ENGINE_DENIES: [&str; 2] = [
    "clippy::wildcard_enum_match_arm",
    "clippy::match_wildcard_for_single_variants",
];

/// The files that may contain unsafe code, each for a reason. Every other
/// Rust file has none (the workspace denies `unsafe_code`, and a file
/// allowing it must be listed here), and every listed file still has some.
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
    (
        "tools/vgbuild/src/ffi.rs",
        "calls the vendored C++ meshoptimizer",
    ),
];

/// Checks the text file `path` (relative, `/`-separated) with text `text`.
pub fn check_text(path: &str, text: &str, problems: &mut Problems) {
    for (index, line) in text.lines().enumerate() {
        for marker in MARKERS {
            if line.match_indices(marker).any(|(at, _)| {
                let before = line[..at].chars().next_back();
                let after = line[at + marker.len()..].chars().next();
                // `MARKERs` counts; `MARKERX` and `aMARKER` are other words.
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

/// Checks the Rust file `path` (relative, `/`-separated) with text
/// `source`, where `lints` are the workspace's.
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
            // Above the block, or above the statement it is part of.
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

/// The engine's error for games, and the file of every engine error's
/// conversion into it.
const ENGINE_ERROR: &str = "crates/engine/src/error.rs";

/// A public error type an engine crate declares: where, and its name.
pub struct ErrorType {
    path: String,
    line: usize,
    name: String,
}

/// Collects the public error types (`pub struct` or `pub enum` named
/// `*Error`) the engine crate file `path` declares in `source`; the engine's
/// own error is not one.
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

/// Fails at each of `found` that does not convert into the engine's error:
/// an `impl From<…Name> for Error` in [`ENGINE_ERROR`].
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

/// Whether the workspace lint `key` is `lint`, or a tool's whole group
/// (`rustdoc::all`) that holds it.
fn covers(key: &str, lint: &str) -> bool {
    key == lint
        || key
            .strip_suffix("all")
            .is_some_and(|tool| tool.ends_with("::") && lint.starts_with(tool))
}

/// Whether `path` is an engine crate's library root (`crates/*/src/lib.rs`).
fn engine_library(path: &str) -> bool {
    path.strip_prefix("crates/")
        .and_then(|rest| rest.strip_suffix("/src/lib.rs"))
        .is_some_and(|name| !name.contains('/'))
}

/// Whether `path` is the root file of a crate target (a library, binary,
/// example or build script), where inner attributes apply crate-wide.
fn crate_root(path: &str) -> bool {
    let (folder, name) = path.rsplit_once('/').unwrap_or(("", path));
    matches!(name, "lib.rs" | "main.rs" | "build.rs")
        || folder.ends_with("/bin")
        || folder.ends_with("/examples")
}
