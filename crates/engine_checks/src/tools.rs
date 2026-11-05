use std::fs;
use std::path::Path;
use std::process::Command;

use crate::Repository;

const RUSTFMT_BATCH: usize = 200;

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

pub fn clippy(repository: &Repository, packages: &[&str], options: &[&str]) -> Result<(), String> {
    let root = repository.root.as_path();
    let (free, strict): (Vec<&str>, Vec<&str>) = packages
        .iter()
        .copied()
        .partition(|package| repository.free(package));
    if !strict.is_empty() {
        let mut arguments = vec!["clippy"];
        for package in &strict {
            arguments.extend(["-p", *package]);
        }
        arguments.extend(options);
        arguments.extend(["--locked", "--", "--no-deps", "-D", "warnings"]);
        run(root, "cargo", &arguments)?;
    }
    clippy_free(root, &free, options)
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

pub fn fmt(repository: &Repository) -> Result<(), String> {
    let root = repository.root.as_path();
    if repository.is_workspace_root() {
        run(root, "cargo", &["fmt", "--all", "--", "--check"])?;
    } else {
        let mut arguments = vec!["fmt".to_owned()];
        for package in repository.own() {
            arguments.extend(["-p".to_owned(), package.name.clone()]);
        }
        arguments.extend(["--".to_owned(), "--check".to_owned()]);
        let arguments: Vec<&str> = arguments.iter().map(String::as_str).collect();
        run(root, "cargo", &arguments)?;
    }
    let edition = edition(&repository.workspace_root)?;
    let files: Vec<String> = crate::files::sources(root)?
        .into_iter()
        .filter(|path| path.ends_with(".rs"))
        .collect();
    let mut unformatted = Vec::new();
    for batch in files.chunks(RUSTFMT_BATCH) {
        let output = Command::new("rustfmt")
            .args([
                "--edition",
                edition.as_str(),
                "--check",
                "--files-with-diff",
            ])
            .args(batch)
            .current_dir(root)
            .output()
            .map_err(|error| format!("cannot start rustfmt: {error}"))?;
        let listed = String::from_utf8_lossy(&output.stdout);
        let before = unformatted.len();
        unformatted.extend(
            listed
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(|line| line.trim_start_matches(r"\\?\").to_owned()),
        );
        if !output.status.success() && unformatted.len() == before {
            return Err(format!(
                "rustfmt could not check its files: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
    }
    if unformatted.is_empty() {
        println!(
            "{}",
            serde_json::json!({"event": "rustfmt_formatted", "files": files.len()})
        );
        return Ok(());
    }
    Err(format!(
        "rustfmt would reformat {} file(s) cargo fmt does not reach (modules a macro \
         declares): run rustfmt --edition {edition} on\n{}",
        unformatted.len(),
        unformatted.join("\n")
    ))
}

fn edition(workspace_root: &Path) -> Result<String, String> {
    let manifest = fs::read_to_string(workspace_root.join("Cargo.toml"))
        .map_err(|error| format!("cannot read Cargo.toml: {error}"))?;
    manifest
        .split_once("[workspace.package]")
        .map(|(_, rest)| rest.split("\n[").next().unwrap_or(rest))
        .and_then(|table| {
            table
                .lines()
                .find_map(|line| line.trim().strip_prefix("edition = "))
        })
        .map(|value| value.trim().trim_matches('"').to_owned())
        .ok_or_else(|| "Cargo.toml's [workspace.package] declares no edition".to_owned())
}
