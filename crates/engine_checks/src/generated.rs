use std::fs;
use std::path::Path;

pub fn write(root: &Path, files: &[(String, String)]) -> Result<(), String> {
    for (path, contents) in files {
        if let Some(directory) = Path::new(path).parent() {
            fs::create_dir_all(root.join(directory))
                .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
        }
        let target = root.join(path);
        let event = if fs::read_to_string(&target).is_ok_and(|held| same_text(&held, contents)) {
            "unchanged"
        } else {
            fs::write(target, contents).map_err(|error| format!("cannot write {path}: {error}"))?;
            "generated"
        };
        println!("{}", serde_json::json!({"event": event, "file": path}));
    }
    Ok(())
}

fn same_text(first: &str, second: &str) -> bool {
    first.replace("\r\n", "\n") == second.replace("\r\n", "\n")
}
