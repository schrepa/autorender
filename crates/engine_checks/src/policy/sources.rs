use engine_core::{Error, ErrorCode};

use super::{Problems, rust};
use crate::{Errors, Rules, at};

pub fn lockfile(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    name == "Cargo.lock" || name == "package-lock.json"
}

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
                problems.push(Error::new(ErrorCode::Marker, at(path, index + 1)).got(marker));
            }
        }
    }
}

pub fn check(
    path: &str,
    source: &str,
    lints: &[String],
    libraries: &[String],
    rules: &Rules,
    problems: &mut Problems,
) {
    let split = rust::split(source);
    if crate_root(path) {
        let code: String = split.code.concat().split_whitespace().collect();
        for attribute in ["#![allow(", "#![expect("] {
            for (at, _) in code.match_indices(attribute) {
                let listed = &code[at + attribute.len()..];
                let listed = &listed[..listed.find(')').unwrap_or(listed.len())];
                for lint in listed.split(',').filter(|item| !item.contains('=')) {
                    if lints.iter().any(|key| covers(key, lint)) || GROUPS.contains(&lint) {
                        problems.push(Error::new(ErrorCode::LintAllowed, path).got(lint));
                    }
                }
            }
        }
        if libraries.iter().any(|library| library == path) {
            let denied: Vec<&str> = code
                .match_indices("#![deny(")
                .flat_map(|(at, attribute)| {
                    let listed = &code[at + attribute.len()..];
                    listed[..listed.find(')').unwrap_or(listed.len())].split(',')
                })
                .collect();
            for lint in ENGINE_DENIES.iter().filter(|lint| !denied.contains(lint)) {
                problems.push(Error::new(ErrorCode::LintMissing, path).got(lint));
            }
        }
    }
    let listed = rules.exceptions.unsafe_files.contains(&path);
    let first = split
        .code
        .iter()
        .position(|code| rust::has_word(code, "unsafe") || rust::has_word(code, "unsafe_code"));
    if !listed && let Some(line) = first {
        problems.push(Error::new(ErrorCode::UnsafeUnlisted, at(path, line + 1)));
    }
    if listed && !split.code.iter().any(|code| rust::has_word(code, "unsafe")) {
        problems.push(Error::new(ErrorCode::ExceptionUnused, at(path, 1)).got("unsafe_files"));
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
                problems.push(Error::new(ErrorCode::SafetyMissing, at(path, index + 1)).got(kind));
            }
        }
    }
}

pub fn error_types(path: &str, source: &str, errors: &Errors, problems: &mut Problems) {
    if !path.starts_with(errors.within) || path == errors.file {
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
            problems.push(
                Error::new(ErrorCode::ErrorType, at(path, index + 1))
                    .got(name)
                    .nearest(errors.file),
            );
        }
    }
}

fn covers(key: &str, lint: &str) -> bool {
    key == lint
        || key
            .strip_suffix("all")
            .is_some_and(|tool| tool.ends_with("::") && lint.starts_with(tool))
}

fn crate_root(path: &str) -> bool {
    let (folder, name) = path.rsplit_once('/').unwrap_or(("", path));
    matches!(name, "lib.rs" | "main.rs" | "build.rs")
        || folder.ends_with("/bin")
        || folder.ends_with("/examples")
}
