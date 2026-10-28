use std::fs;
use std::path::Path;
use std::process::Command;

const LEAVES: &str = "examples";

const REACHING: [&str; 2] = ["crates", "tools"];

pub fn check(root: &Path) -> Result<(), String> {
    let packages = packages(root)?;
    let needle = format!("{LEAVES}/");
    let mut problems = Vec::new();
    let files = files(root)?;
    for path in &files {
        let Ok(text) = fs::read(root.join(path)) else {
            continue;
        };
        let Ok(text) = String::from_utf8(text) else {
            continue;
        };
        let manifest = path.ends_with("Cargo.toml");
        let mut dependencies = false;
        for (index, line) in text.lines().enumerate() {
            let at = format!("{path}:{}", index + 1);
            for (start, _) in line.match_indices(&needle) {
                if reaches_root(&line[..start]) {
                    problems.push(format!("{at}: example_named {}", line.trim()));
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
    if problems.is_empty() {
        println!(
            "{} files under {} reach no package of {LEAVES}/ ({}).",
            files.len(),
            REACHING.join(", "),
            packages.join(", ")
        );
        return Ok(());
    }
    Err(problems.join("\n"))
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

fn files(root: &Path) -> Result<Vec<String>, String> {
    let output = Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
        ])
        .args(REACHING)
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot run git ls-files: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git ls-files failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let mut files: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(str::to_owned)
        .collect();
    files.sort();
    files.dedup();
    Ok(files)
}

fn packages(root: &Path) -> Result<Vec<String>, String> {
    let folder = root.join(LEAVES);
    let mut packages = Vec::new();
    if !folder.is_dir() {
        return Ok(packages);
    }
    let entries =
        fs::read_dir(&folder).map_err(|error| format!("cannot read {LEAVES}: {error}"))?;
    for entry in entries {
        let manifest = entry.map_err(|error| error.to_string())?.path().join("Cargo.toml");
        let Ok(text) = fs::read_to_string(&manifest) else {
            continue;
        };
        let document = toml_edit::Document::parse(text)
            .map_err(|error| format!("{}: {error}", manifest.display()))?;
        if let Some(name) = document
            .as_item()
            .get("package")
            .and_then(|package| package.get("name"))
            .and_then(toml_edit::Item::as_str)
        {
            packages.push(name.to_owned());
        }
    }
    packages.sort();
    Ok(packages)
}
