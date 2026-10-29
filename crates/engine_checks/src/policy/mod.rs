mod manifests;
pub mod rust;
mod sources;

use crate::{Repository, Rules};

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

pub fn check(repository: &Repository, rules: &Rules) -> Result<(), String> {
    let root = repository.root.as_path();
    let files: Vec<String> = crate::files::listed(root, &[])?
        .into_iter()
        .filter(|path| !path.starts_with("vendor/") && root.join(path).is_file())
        .collect();
    let mut problems = Problems::default();
    let package_jsons: Vec<String> = files
        .iter()
        .filter(|path| path.ends_with("package.json") && !path.contains("node_modules/"))
        .cloned()
        .collect();
    let lints = manifests::check(repository, &package_jsons, &mut problems)?;
    let runtime = repository.runtime();
    let libraries: Vec<String> = repository
        .own()
        .filter(|package| runtime.contains(package.name.as_str()))
        .filter(|package| {
            package
                .path
                .as_deref()
                .and_then(|path| path.strip_prefix("crates/"))
                .is_some_and(|name| !name.contains('/'))
        })
        .filter_map(|package| package.file("src/lib.rs"))
        .collect();
    let mut errors = Vec::new();
    for path in &files {
        if sources::lockfile(path) {
            continue;
        }
        let bytes = std::fs::read(root.join(path))
            .map_err(|error| format!("cannot read {path}: {error}"))?;
        let Ok(text) = String::from_utf8(bytes) else {
            continue;
        };
        sources::check_text(path, &text, &mut problems);
        if path.ends_with(".rs") {
            sources::check(path, &text, &lints, &libraries, rules, &mut problems);
            if let Some(conversions) = &rules.errors {
                sources::error_types(path, &text, conversions, &mut errors);
            }
        }
    }
    if let Some(conversions) = &rules.errors {
        sources::conversions(root, &errors, conversions, &mut problems)?;
    }
    for path in rules.exceptions.unsafe_files {
        if !files.iter().any(|file| file == path) {
            problems.file(
                path,
                "listed in the exceptions' unsafe_files but does not exist: remove it from the \
                 list",
            );
        }
    }
    if problems.0.is_empty() {
        println!("Policy holds.");
        return Ok(());
    }
    Err(format!(
        "{} policy problem(s):\n{}",
        problems.0.len(),
        problems.0.join("\n")
    ))
}
