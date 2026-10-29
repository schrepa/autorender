use std::fs;
use std::path::Path;

pub fn write(root: &Path, files: &[(String, String)]) -> Result<(), String> {
    for (path, contents) in files {
        if let Some(directory) = Path::new(path).parent() {
            fs::create_dir_all(root.join(directory))
                .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
        }
        fs::write(root.join(path), contents)
            .map_err(|error| format!("cannot write {path}: {error}"))?;
        println!("Generated {path}.");
    }
    Ok(())
}

pub fn check(root: &Path, files: &[(String, String)], schema: &str) -> Result<(), String> {
    let mut stale = Vec::new();
    for (path, contents) in files {
        match fs::read_to_string(root.join(path)) {
            Ok(saved) if saved.replace("\r\n", "\n") == *contents => {
                println!("Generated file matches {path}.");
            }
            Ok(_) => stale.push(format!("{path} is out of date")),
            Err(error) => stale.push(format!("cannot read {path}: {error}")),
        }
    }
    if stale.is_empty() {
        return Ok(());
    }
    Err(format!("{}; run `{schema}`", stale.join("; ")))
}
