use std::path::Path;
use std::process::Command;

pub fn run(root: &Path, program: &str, arguments: &[&str]) -> Result<(), String> {
    let record = serde_json::json!({
        "event": "running",
        "program": program,
        "arguments": arguments,
    });
    println!("{record}");
    let status = Command::new(program)
        .args(arguments)
        .current_dir(root)
        .status()
        .map_err(|error| {
            format!("cannot start {program}: {error}; ensure it is installed and available on PATH")
        })?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "{program} {} failed ({status}); see the command output above",
            arguments.join(" ")
        ))
    }
}

pub fn clippy_free(root: &Path, packages: &[&str], options: &[&str]) -> Result<(), String> {
    if packages.is_empty() {
        return Ok(());
    }
    let mut arguments = vec!["clippy"];
    for package in packages {
        arguments.extend(["-p", *package]);
    }
    arguments.extend(options);
    arguments.extend(["--locked", "--", "--no-deps"]);
    run(root, "cargo", &arguments)
}
