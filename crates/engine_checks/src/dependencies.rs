use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use engine_core::{Error, ErrorCode};
use serde_json::json;

use crate::bricks::Recorded;
use crate::faces::{Face, parse};
use crate::{DependencyKind, Exceptions, Repository};

pub fn check(repository: &Repository, exceptions: &Exceptions) -> Result<(), String> {
    let recorded = crate::bricks::recorded(repository)?;
    let bricks = crate::bricks::brick_crates(
        repository,
        recorded.iter().map(|brick| brick.krate.as_str()),
    );
    let mut problems = Vec::new();
    let used = core_uses(repository, &bricks, exceptions, &mut problems)?;
    for &(from, to) in exceptions.core_uses_brick {
        let Some(package) = repository.package(from).filter(|package| package.own()) else {
            continue;
        };
        if !used.contains(&(from.to_owned(), to.to_owned())) {
            let file = repository.relative(&package.manifest).unwrap_or_default();
            problems.push(
                Error::new(ErrorCode::ExceptionUnused, file)
                    .got(json!({"core_uses_brick": [from, to]})),
            );
        }
    }
    internal_paths(repository, &bricks, &mut problems)?;
    brick_cycles(&recorded, &mut problems);
    crate_cycles(repository, &mut problems);
    if problems.is_empty() {
        println!(
            "{}",
            json!({
                "event": "dependencies_hold",
                "bricks": bricks,
                "exceptions": exceptions.core_uses_brick.len(),
            })
        );
        return Ok(());
    }
    crate::report(&problems)
}

fn core_uses(
    repository: &Repository,
    bricks: &BTreeSet<String>,
    exceptions: &Exceptions,
    problems: &mut Vec<Error>,
) -> Result<BTreeSet<(String, String)>, String> {
    let mut used = BTreeSet::new();
    for package in repository.own().filter(|package| package.core()) {
        let manifest = fs::read_to_string(&package.manifest)
            .map_err(|error| format!("cannot read {}: {error}", package.manifest.display()))?;
        let file = repository.relative(&package.manifest).unwrap_or_default();
        let mut reported = BTreeSet::new();
        for dependency in &package.dependencies {
            if dependency.kind == DependencyKind::Development {
                continue;
            }
            for (brick, through) in reached(repository, &dependency.package, bricks) {
                let excepted = exceptions
                    .core_uses_brick
                    .iter()
                    .any(|&(from, to)| from == package.name && to == brick);
                if excepted {
                    used.insert((package.name.clone(), brick));
                    continue;
                }
                if !reported.insert(brick.clone()) {
                    continue;
                }
                let line = line_of(&manifest, &dependency.package);
                problems.push(
                    Error::new(ErrorCode::CoreUsesBrick, crate::at(&file, line)).got(json!({
                        "from": package.name,
                        "to": brick,
                        "through": through,
                    })),
                );
            }
        }
    }
    Ok(used)
}

fn reached(
    repository: &Repository,
    start: &str,
    bricks: &BTreeSet<String>,
) -> Vec<(String, Vec<String>)> {
    let mut found = Vec::new();
    let mut seen = BTreeSet::new();
    let mut pending = vec![(start.to_owned(), Vec::new())];
    while let Some((name, through)) = pending.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let Some(package) = repository.package(&name) else {
            continue;
        };
        if bricks.contains(&package.crate_name()) {
            found.push((name, through));
            continue;
        }
        let walked = package.member || !package.engine.is_null();
        if package.core() || !walked {
            continue;
        }
        let mut next = through.clone();
        next.push(name.clone());
        for dependency in &package.dependencies {
            if dependency.kind != DependencyKind::Development {
                pending.push((dependency.package.clone(), next.clone()));
            }
        }
    }
    found
}

fn line_of(manifest: &str, name: &str) -> usize {
    manifest
        .lines()
        .position(|line| {
            let line = line.trim();
            line.strip_prefix(name)
                .is_some_and(|rest| rest.starts_with(['.', ' ', '=']))
                || line.ends_with(&format!("dependencies.{name}]"))
        })
        .map_or(0, |index| index + 1)
}

fn internal_paths(
    repository: &Repository,
    bricks: &BTreeSet<String>,
    problems: &mut Vec<Error>,
) -> Result<(), String> {
    let mut faces: BTreeMap<String, Face> = BTreeMap::new();
    for package in repository.own() {
        let own = package.crate_name();
        if !bricks.contains(&own) {
            continue;
        }
        let mut files = Vec::new();
        rust_files(&package.folder.join("src"), &mut files);
        files.sort();
        for file in files {
            let source = fs::read_to_string(&file)
                .map_err(|error| format!("cannot read {}: {error}", file.display()))?;
            let code = parse::code(&source);
            let shown = repository
                .relative(&file)
                .unwrap_or_else(|| file.display().to_string());
            for other in bricks.iter().filter(|other| **other != own) {
                for (line, segments) in paths(&code, other) {
                    if !faces.contains_key(other) {
                        let Some(found) = repository.of_crate(other) else {
                            continue;
                        };
                        faces.insert(other.clone(), crate::faces::face(&found.folder)?);
                    }
                    if faces.get(other).is_none_or(|face| face.reaches(&segments)) {
                        continue;
                    }
                    problems.push(
                        Error::new(ErrorCode::BrickInternalPath, crate::at(&shown, line)).got(
                            json!({
                                "from": own,
                                "to": other,
                                "path": format!("{other}::{}", segments.join("::")),
                            }),
                        ),
                    );
                }
            }
        }
    }
    Ok(())
}

fn rust_files(folder: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(folder).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}

fn segments_of(path: &str) -> Vec<String> {
    path.split("::")
        .filter(|segment| !segment.is_empty())
        .map(str::to_owned)
        .collect()
}

fn tree(
    code: &[char],
    open: usize,
    prefix: &[String],
    line: usize,
) -> (Vec<(usize, Vec<String>)>, usize) {
    let close = parse::closing(code, open);
    let text: String = code[open..close.min(code.len())].iter().collect();
    let found = parse::expand(&text)
        .into_iter()
        .map(|(path, _)| {
            let mut segments = prefix.to_vec();
            segments.extend(segments_of(&path));
            (line, segments)
        })
        .collect();
    (found, close)
}

fn paths(code: &[char], krate: &str) -> Vec<(usize, Vec<String>)> {
    let needle: Vec<char> = format!("{krate}::").chars().collect();
    let mut found = Vec::new();
    let mut at = 0;
    while at + needle.len() <= code.len() {
        let before = at.checked_sub(1).map(|index| code[index]);
        if code[at..at + needle.len()] != needle[..]
            || before.is_some_and(|character| parse::identifier(character) || character == ':')
        {
            at += 1;
            continue;
        }
        let line = code[..at]
            .iter()
            .filter(|&&character| character == '\n')
            .count()
            + 1;
        let mut end = at + needle.len();
        let mut segments = Vec::new();
        let mut expanded = false;
        loop {
            if code.get(end) == Some(&'{') {
                let (paths, close) = tree(code, end, &segments, line);
                found.extend(paths);
                expanded = true;
                end = close;
                break;
            }
            let name: String = code[end..]
                .iter()
                .take_while(|&&character| parse::identifier(character))
                .collect();
            if name.is_empty() {
                break;
            }
            end += name.chars().count();
            segments.push(name);
            if code.get(end) == Some(&':')
                && code.get(end + 1) == Some(&':')
                && !matches!(code.get(end + 2), Some('*' | '<'))
            {
                end += 2;
                continue;
            }
            break;
        }
        if !expanded && !segments.is_empty() {
            found.push((line, segments));
        }
        at = end.max(at + 1);
    }
    found
}

fn brick_cycles(recorded: &[Recorded], problems: &mut Vec<Error>) {
    let graph: BTreeMap<String, Vec<String>> = recorded
        .iter()
        .map(|brick| (brick.name.clone(), brick.reads.clone()))
        .collect();
    for cycle in cycles(&graph) {
        problems.push(Error::new(ErrorCode::BrickCycle, crate::bricks::FILE).got(cycle));
    }
}

fn crate_cycles(repository: &Repository, problems: &mut Vec<Error>) {
    let own: BTreeSet<&str> = repository
        .own()
        .map(|package| package.name.as_str())
        .collect();
    let graph: BTreeMap<String, Vec<String>> = repository
        .own()
        .map(|package| {
            let mut targets: Vec<String> = package
                .dependencies
                .iter()
                .map(|dependency| dependency.package.clone())
                .filter(|name| own.contains(name.as_str()))
                .collect();
            targets.sort();
            targets.dedup();
            (package.name.clone(), targets)
        })
        .collect();
    for cycle in cycles(&graph) {
        problems.push(Error::new(ErrorCode::CrateCycle, "Cargo.toml").got(cycle));
    }
}

fn cycles(graph: &BTreeMap<String, Vec<String>>) -> Vec<Vec<String>> {
    let mut found = Vec::new();
    let mut done = BTreeSet::new();
    for start in graph.keys() {
        let mut stack = vec![start.clone()];
        visit(graph, start, &mut stack, &mut done, &mut found);
    }
    found
}

fn visit(
    graph: &BTreeMap<String, Vec<String>>,
    node: &str,
    stack: &mut Vec<String>,
    done: &mut BTreeSet<String>,
    found: &mut Vec<Vec<String>>,
) {
    if done.contains(node) {
        return;
    }
    for next in graph.get(node).into_iter().flatten() {
        if let Some(position) = stack.iter().position(|seen| seen == next) {
            let mut cycle = stack[position..].to_vec();
            cycle.push(next.clone());
            found.push(cycle);
            continue;
        }
        if done.contains(next) {
            continue;
        }
        stack.push(next.clone());
        visit(graph, next, stack, done, found);
        stack.pop();
    }
    done.insert(node.to_owned());
}
