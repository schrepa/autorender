mod manifests;
pub mod rust;
mod sources;

use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Default)]
pub struct Problems(Vec<String>);

impl Problems {
    fn at(&mut self, path: &str, line: usize, message: &str) {
        self.0.push(format!("{path}:{line}: {message}"));
    }

    fn file(&mut self, path: &str, message: &str) {
        self.0.push(format!("{path}: {message}"));
    }
}

pub fn repository_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| "cannot locate the workspace above tools/policy".to_owned())
}

fn files(root: &Path) -> Result<Vec<String>, String> {
    let output = Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
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
        .filter(|path| !path.is_empty() && !path.starts_with("vendor/"))
        .filter(|path| root.join(path).is_file())
        .map(str::to_owned)
        .collect();
    files.sort();
    files.dedup();
    Ok(files)
}

pub fn check(root: &Path) -> Result<(), String> {
    let files = files(root)?;
    let mut problems = Problems::default();
    let package_jsons: Vec<String> = files
        .iter()
        .filter(|path| path.ends_with("package.json") && !path.contains("node_modules/"))
        .cloned()
        .collect();
    let lints = manifests::check(root, &package_jsons, &mut problems)?;
    let mut errors = Vec::new();
    for path in &files {
        if sources::LOCKFILES.contains(&path.as_str()) {
            continue;
        }
        let bytes = std::fs::read(root.join(path))
            .map_err(|error| format!("cannot read {path}: {error}"))?;
        let Ok(text) = String::from_utf8(bytes) else {
            continue;
        };
        sources::check_text(path, &text, &mut problems);
        if path.ends_with(".rs") {
            sources::check(path, &text, &lints, &mut problems);
            sources::error_types(path, &text, &mut errors);
        }
    }
    sources::conversions(root, &errors, &mut problems)?;
    for (path, _) in sources::UNSAFE_FILES {
        if !files.iter().any(|file| file == path) {
            problems.file(
                path,
                "listed in UNSAFE_FILES (tools/policy/src/sources.rs) but does not exist: remove \
                 it from the list",
            );
        }
    }
    if problems.0.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{} policy problem(s):\n{}",
        problems.0.len(),
        problems.0.join("\n")
    ))
}
