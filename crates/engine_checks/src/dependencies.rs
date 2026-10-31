use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use engine::Engine;
use engine::bricks::BrickRegistry;
use engine_core::{Error, ErrorCode, Place};
use serde_json::json;

use crate::bricks::Recorded;
use crate::faces::{Face, parse};
use crate::{DependencyKind, Exceptions, Repository};

const GPU_PART: &str = "GpuPart for ";

pub fn check(
    repository: &Repository,
    exceptions: &Exceptions,
    engine: &Engine,
) -> Result<(), String> {
    let recorded = crate::bricks::recorded(repository)?;
    let bricks = crate::bricks::brick_crates(
        repository,
        recorded.iter().map(|brick| brick.krate.as_str()),
    );
    let registry = engine.app.world().get_resource::<BrickRegistry>();
    let places: Vec<Place> = registry.map(BrickRegistry::places).unwrap_or_default();
    let faces = face_crates(repository, &places, &bricks, |_| true);
    let gpu_faces = face_crates(repository, &places, &bricks, |face| {
        face.gpu.is_some() || face.wgsl.is_some()
    });
    let gpu = gpu_crates(repository, &recorded);
    let mut problems = Vec::new();
    let reached: BTreeSet<String> = bricks.union(&gpu_faces).cloned().collect();
    let used = core_uses(repository, &reached, exceptions, &mut problems)?;
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
    gpu_dependencies(repository, &gpu, &mut problems)?;
    face_crates_hold_no_bricks(repository, &faces, registry, &mut problems)?;
    brick_cycles(&recorded, &mut problems);
    crate_cycles(repository, &mut problems);
    if problems.is_empty() {
        println!(
            "{}",
            json!({
                "event": "dependencies_hold",
                "bricks": bricks,
                "gpu_bricks": gpu,
                "faces": faces,
                "exceptions": exceptions.core_uses_brick.len(),
            })
        );
        return Ok(());
    }
    crate::report(&problems)
}

fn face_crates(
    repository: &Repository,
    places: &[Place],
    bricks: &BTreeSet<String>,
    sided: fn(&engine_core::Face) -> bool,
) -> BTreeSet<String> {
    places
        .iter()
        .filter(|place| sided(&place.face))
        .map(|place| place.declared_in().to_owned())
        .filter(|name| !bricks.contains(name))
        .filter(|name| {
            repository
                .of_crate(name)
                .is_some_and(|package| !package.core())
        })
        .collect()
}

fn gpu_crates(repository: &Repository, recorded: &[Recorded]) -> BTreeSet<String> {
    let mut crates: BTreeSet<String> = recorded
        .iter()
        .filter(|brick| !brick.parts.is_empty())
        .map(|brick| brick.krate.clone())
        .collect();
    let replacing: Vec<String> = repository
        .own()
        .filter(|package| package.replaces().is_some_and(|name| crates.contains(name)))
        .map(crate::Package::crate_name)
        .collect();
    crates.extend(replacing);
    crates
}

fn gpu_dependencies(
    repository: &Repository,
    gpu: &BTreeSet<String>,
    problems: &mut Vec<Error>,
) -> Result<(), String> {
    for package in repository
        .own()
        .filter(|package| gpu.contains(&package.crate_name()))
    {
        let manifest = fs::read_to_string(&package.manifest)
            .map_err(|error| format!("cannot read {}: {error}", package.manifest.display()))?;
        let file = repository.relative(&package.manifest).unwrap_or_default();
        for dependency in &package.dependencies {
            if dependency.kind == DependencyKind::Development {
                continue;
            }
            let Some(other) = repository.package(&dependency.package) else {
                continue;
            };
            if other.name == package.name || !gpu.contains(&other.crate_name()) {
                continue;
            }
            let line = line_of(&manifest, &dependency.package);
            problems.push(
                Error::new(ErrorCode::BrickInternalPath, crate::at(&file, line)).got(json!({
                    "from": package.name,
                    "to": other.name,
                })),
            );
        }
    }
    Ok(())
}

fn face_crates_hold_no_bricks(
    repository: &Repository,
    faces: &BTreeSet<String>,
    registry: Option<&BrickRegistry>,
    problems: &mut Vec<Error>,
) -> Result<(), String> {
    for name in faces {
        let Some(package) = repository.of_crate(name).filter(|package| package.own()) else {
            continue;
        };
        let file = repository.relative(&package.manifest).unwrap_or_default();
        if let Some(brick) = registry
            .into_iter()
            .flat_map(BrickRegistry::bricks)
            .find(|brick| brick.crate_name == name.as_str())
        {
            problems.push(
                Error::new(ErrorCode::BrickInternalPath, file.as_str())
                    .got(json!({ "faces": name, "brick": brick.name })),
            );
        }
        let mut files = Vec::new();
        rust_files(&package.folder.join("src"), &mut files);
        files.sort();
        for path in files {
            let source = fs::read_to_string(&path)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            let code: String = parse::code(&source).into_iter().collect();
            if let Some(at) = code.find(GPU_PART) {
                let line = code[..at].matches('\n').count() + 1;
                let shown = repository
                    .relative(&path)
                    .unwrap_or_else(|| path.display().to_string());
                problems.push(
                    Error::new(ErrorCode::BrickInternalPath, crate::at(&shown, line))
                        .got(json!({ "faces": name, "holds": "GpuPart" })),
                );
            }
        }
    }
    Ok(())
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

pub(crate) fn rust_files(folder: &Path, files: &mut Vec<PathBuf>) {
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
