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
        println!(
            "{}",
            serde_json::json!({"event": "generated", "file": path})
        );
    }
    Ok(())
}
