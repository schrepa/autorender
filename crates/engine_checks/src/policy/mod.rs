mod manifests;
pub mod rust;
mod sources;

use engine_core::{Error, ErrorCode};

use crate::{Repository, Rules};

#[derive(Default)]
pub struct Problems(Vec<Error>);

impl Problems {
    fn push(&mut self, error: Error) {
        self.0.push(error);
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
            if let Some(errors) = &rules.errors {
                sources::error_types(path, &text, errors, &mut problems);
            }
        }
    }
    for path in rules.exceptions.unsafe_files {
        if !files.iter().any(|file| file == path) {
            problems.push(Error::new(ErrorCode::ExceptionUnused, *path).got("unsafe_files"));
        }
    }
    if problems.0.is_empty() {
        println!("{}", serde_json::json!({"event": "policy_holds"}));
        return Ok(());
    }
    crate::report(&problems.0)
}
