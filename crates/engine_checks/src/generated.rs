use std::fs;
use std::path::Path;

use engine_core::{Error, ErrorCode};

pub fn write(root: &Path, files: &[(String, String)]) -> Result<(), String> {
    for (path, contents) in files {
        if let Some(directory) = Path::new(path).parent() {
            fs::create_dir_all(root.join(directory))
                .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
        }
        fs::write(root.join(path), contents)
            .map_err(|error| format!("cannot write {path}: {error}"))?;
        println!("{}", serde_json::json!({"event": "generated", "file": path}));
    }
    Ok(())
}

pub fn check(root: &Path, files: &[(String, String)], schema: &str) -> Result<(), String> {
    let mut stale = Vec::new();
    for (path, contents) in files {
        match fs::read_to_string(root.join(path)) {
            Ok(saved) if saved.replace("\r\n", "\n") == *contents => {
                println!(
                    "{}",
                    serde_json::json!({"event": "generated_matches", "file": path})
                );
            }
            Ok(_) => {
                stale.push(Error::new(ErrorCode::GeneratedStale, path.as_str()).nearest(schema));
            }
            Err(error) => stale.push(
                Error::new(ErrorCode::SourceIo, path.as_str()).got(format!("{:?}", error.kind())),
            ),
        }
    }
    if stale.is_empty() {
        return Ok(());
    }
    crate::report(&stale)
}
