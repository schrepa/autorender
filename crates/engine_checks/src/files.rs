use std::path::Path;
use std::process::Command;

pub fn listed(root: &Path, within: &[&str]) -> Result<Vec<String>, String> {
    let output = Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
        ])
        .args(within)
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

pub fn ours(path: &str) -> bool {
    !path.starts_with("vendor/") && !path.contains("/generated/") && !path.starts_with("generated/")
}

pub fn sources(root: &Path) -> Result<Vec<String>, String> {
    Ok(listed(root, &[])?
        .into_iter()
        .filter(|path| {
            [".rs", ".wgsl", ".ts", ".mjs"]
                .iter()
                .any(|extension| path.ends_with(extension))
                && ours(path)
                && root.join(path).is_file()
        })
        .collect())
}
