use std::collections::BTreeSet;
use std::fs;

use crate::Repository;

pub fn check(repository: &Repository) -> Result<(), String> {
    let root = repository.root.as_path();
    let (leaves, reaching) = folders(repository);
    let packages: Vec<String> = repository
        .own()
        .filter(|package| {
            package
                .path
                .as_deref()
                .and_then(top)
                .is_some_and(|folder| leaves.contains(folder))
        })
        .map(|package| package.name.clone())
        .collect();
    let needles: Vec<String> = leaves.iter().map(|leaf| format!("{leaf}/")).collect();
    let mut problems = Vec::new();
    let reaching: Vec<&str> = reaching.iter().map(String::as_str).collect();
    let files = crate::files::listed(root, &reaching)?;
    for path in &files {
        let Ok(text) = fs::read(root.join(path)) else {
            continue;
        };
        let Ok(text) = String::from_utf8(text) else {
            continue;
        };
        named(path, &text, &needles, &packages, &mut problems);
    }
    if problems.is_empty() {
        let record = serde_json::json!({
            "event": "leaves_unreached",
            "files": files.len(),
            "reaching": reaching,
            "leaves": leaves,
            "packages": packages,
        });
        println!("{record}");
        return Ok(());
    }
    Err(problems.join("\n"))
}

fn top(path: &str) -> Option<&str> {
    path.split('/').next().filter(|folder| !folder.is_empty())
}

fn folders(repository: &Repository) -> (BTreeSet<String>, BTreeSet<String>) {
    let projects: Vec<&str> = repository
        .projects()
        .filter_map(|project| project.path.as_deref())
        .filter(|path| !path.is_empty())
        .collect();
    let reaching: BTreeSet<String> = repository
        .own()
        .filter_map(|package| package.path.as_deref())
        .filter(|path| {
            !projects
                .iter()
                .any(|project| *path == *project || path.starts_with(&format!("{project}/")))
        })
        .filter_map(top)
        .map(str::to_owned)
        .collect();
    let leaves = projects
        .iter()
        .filter_map(|path| top(path))
        .filter(|folder| !reaching.contains(*folder))
        .map(str::to_owned)
        .collect();
    (leaves, reaching)
}

fn named(
    path: &str,
    text: &str,
    needles: &[String],
    packages: &[String],
    problems: &mut Vec<String>,
) {
    let manifest = path.ends_with("Cargo.toml");
    let mut dependencies = false;
    for (index, line) in text.lines().enumerate() {
        let at = format!("{path}:{}", index + 1);
        for needle in needles {
            for (start, _) in line.match_indices(needle.as_str()) {
                if reaches_root(&line[..start]) {
                    problems.push(format!("{at}: example_named {}", line.trim()));
                }
            }
        }
        if !manifest {
            continue;
        }
        let trimmed = line.trim();
        if let Some(table) = trimmed.strip_prefix('[') {
            let table = table.trim_end_matches(']');
            dependencies = table.ends_with("dependencies");
            if let Some(name) = table
                .split_once("dependencies.")
                .map(|(_, name)| name.trim_matches('"'))
                .filter(|name| packages.iter().any(|package| package == name))
            {
                problems.push(format!("{at}: example_dependency {name}"));
            }
            continue;
        }
        if !dependencies {
            continue;
        }
        let key = trimmed
            .split_once('=')
            .map(|(key, _)| key.trim().split('.').next().unwrap_or_default())
            .unwrap_or_default()
            .trim_matches('"');
        let renamed = trimmed
            .split_once("package")
            .and_then(|(_, rest)| rest.trim_start().strip_prefix('='))
            .and_then(|rest| rest.trim_start().strip_prefix('"'))
            .and_then(|rest| rest.split_once('"'))
            .map(|(name, _)| name);
        for name in [Some(key), renamed].into_iter().flatten() {
            if packages.iter().any(|package| package == name) {
                problems.push(format!("{at}: example_dependency {name}"));
            }
        }
    }
}

fn reaches_root(before: &str) -> bool {
    let Some(folder) = before.strip_suffix('/') else {
        return before
            .chars()
            .next_back()
            .is_none_or(|last| !(last.is_alphanumeric() || last == '_' || last == '-'));
    };
    let segment = folder
        .rsplit(|character: char| character == '/' || !is_path_character(character))
        .next()
        .unwrap_or_default();
    segment.is_empty() || segment.chars().all(|character| character == '.')
}

fn is_path_character(character: char) -> bool {
    character.is_alphanumeric() || matches!(character, '_' | '-' | '.')
}
