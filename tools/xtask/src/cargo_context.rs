//! Reuse registry dependencies across source folders without reusing local
//! outputs from another workspace. Cargo can otherwise call a relocated
//! cdylib fresh when its relative inputs are older than the previous build.

use crate::{cache::Entry, files, jobs};
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

/// Call while holding the exclusive context lease, before any Cargo command.
/// Cargo's public metadata and clean interfaces own the package selection and
/// deletion. No source hash or internal Cargo fingerprint is used here.
pub fn select(
    root: &Path,
    arguments: &[&str],
    context: &Entry,
    target: &Path,
) -> Result<(), String> {
    let routing = routing(arguments)?;
    let located = Command::new("cargo")
        .current_dir(root)
        .args(["locate-project", "--workspace", "--message-format=json"])
        .args(&routing)
        .output()
        .map_err(|e| e.to_string())?;
    if !located.status.success() {
        return Err(String::from_utf8_lossy(&located.stderr).into_owned());
    }
    let located: Value = serde_json::from_slice(&located.stdout).map_err(|e| e.to_string())?;
    let manifest = files::canonical(Path::new(
        located["root"].as_str().ok_or("no Cargo workspace")?,
    ))?;
    fs::create_dir_all(target).map_err(|e| e.to_string())?;
    let selection = json!({"version":2, "manifest":manifest, "target":files::canonical(target)?, "configuration":configuration(root, &routing)?});
    let scope = scope(arguments)?;
    let domain =
        json!({"arguments":scope, "default_target":std::env::var_os("CARGO_BUILD_TARGET")});
    let marker = context.path().join("workspace.json");
    let previous = if marker.is_file() {
        files::json_file(&marker)?
    } else {
        Value::Null
    };
    let mut domains = if previous["selection"] == selection {
        previous["domains"].as_array().cloned().unwrap_or_default()
    } else {
        Vec::new()
    };
    if domains.contains(&domain) {
        return Ok(());
    }
    let logs = context.path().join("workspace-change");
    if logs.exists() {
        files::list(&logs)?;
        fs::remove_dir_all(&logs).map_err(|e| e.to_string())?;
    }
    let mut metadata = Command::new("cargo");
    metadata
        .current_dir(root)
        .args(["metadata", "--format-version=1", "--all-features"])
        .args(&routing)
        .env("CARGO_BUILD_BUILD_DIR", target);
    let metadata = run(&mut metadata, &logs.join("metadata"))?;
    let metadata: Value =
        serde_json::from_slice(&fs::read(metadata.stdout).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    // Include path patches and optional local dependencies, not just members.
    let packages: Vec<_> = metadata["packages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|p| p["source"].is_null())
        .filter_map(|p| p["id"].as_str())
        .collect();
    if !packages.is_empty() {
        let mut clean = Command::new("cargo");
        clean
            .current_dir(root)
            .arg("clean")
            .args(&routing)
            .args(&scope)
            .arg("--target-dir")
            .arg(target)
            .env("CARGO_BUILD_BUILD_DIR", target);
        for package in packages {
            clean.args(["-p", package]);
        }
        pin_build_directory(&mut clean, target)?;
        run(&mut clean, &logs.join("clean"))?;
    }
    // A failure leaves the old marker in place, forcing cleanup on retry.
    domains.push(domain);
    files::atomic_json(&marker, &json!({"selection":selection, "domains":domains}))
}

/// Keep intermediate outputs under the leased target even with a caller's
/// build-dir configuration. Cargo command-line configuration wins over env.
pub fn pin_build_directory(command: &mut Command, target: &Path) -> Result<(), String> {
    let path = serde_json::to_string(target).map_err(|e| e.to_string())?;
    command
        .arg("--config")
        .arg(format!("build.build-dir={path}"));
    Ok(())
}

// A source-folder switch invalidates every profile/target domain. Clean each
// on its next use: `cargo clean -p` alone covers neither custom profiles nor
// explicit cross targets. Cargo applies the configured default target itself.
fn scope(arguments: &[&str]) -> Result<Vec<String>, String> {
    let mut profile = "dev";
    let mut targets = Vec::new();
    let mut args = arguments.iter().copied();
    while let Some(arg) = args.next() {
        match arg {
            "--profile" => profile = args.next().ok_or("missing Cargo profile")?,
            "--release" | "-r" => profile = "release",
            "--target" => targets.extend([
                "--target".to_owned(),
                args.next().ok_or("missing Cargo target")?.to_owned(),
            ]),
            _ => {
                if let Some(value) = arg.strip_prefix("--profile=") {
                    profile = value;
                }
                if let Some(value) = arg.strip_prefix("--target=") {
                    targets.extend(["--target".to_owned(), value.to_owned()]);
                }
            }
        }
    }
    targets.extend(["--profile".to_owned(), profile.to_owned()]);
    Ok(targets)
}

// Configuration can change the implicit target. It only invalidates the
// sharing boundary; Cargo still decides source freshness on every invocation.
fn configuration(root: &Path, routing: &[&str]) -> Result<Value, String> {
    let mut directories: Vec<_> = root.ancestors().map(|p| p.join(".cargo")).collect();
    let home = std::env::var_os("CARGO_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("USERPROFILE")
                .or_else(|| std::env::var_os("HOME"))
                .map(|p| Path::new(&p).join(".cargo"))
        });
    directories.extend(home);
    let mut paths: Vec<_> = directories
        .iter()
        .flat_map(|p| [p.join("config"), p.join("config.toml")])
        .collect();
    let mut overrides = Vec::new();
    let mut args = routing.iter().copied();
    while let Some(arg) = args.next() {
        let value = if arg == "--config" {
            args.next()
        } else {
            arg.strip_prefix("--config=")
        };
        if let Some(value) = value {
            overrides.push(value);
            if !value.contains('=') {
                paths.push(root.join(value));
            }
        }
    }
    let files = paths
        .iter()
        .filter(|p| p.is_file())
        .map(|p| Ok(json!({"path":p, "sha256":files::hash(p)?})))
        .collect::<Result<Vec<_>, String>>()?;
    Ok(json!({"files":files, "overrides":overrides}))
}

fn run(command: &mut Command, logs: &Path) -> Result<jobs::Output, String> {
    let output = jobs::run(command, logs, &jobs::Limits::default())?;
    if !output.status.success() {
        return Err(format!(
            "Cargo workspace selection failed: {}; {}",
            fs::read_to_string(&output.stderr).unwrap_or_default(),
            logs.display()
        ));
    }
    Ok(output)
}

fn routing<'a>(arguments: &[&'a str]) -> Result<Vec<&'a str>, String> {
    let mut result = Vec::new();
    let mut args = arguments.iter().copied();
    while let Some(arg) = args.next() {
        if ["--manifest-path", "-m", "--config", "-Z"].contains(&arg) {
            result.extend([arg, args.next().ok_or("missing Cargo option value")?]);
        } else if ["--locked", "--offline", "--frozen"].contains(&arg)
            || ["--manifest-path=", "--config=", "-Z"]
                .iter()
                .any(|p| arg.starts_with(p))
        {
            result.push(arg);
        }
    }
    Ok(result)
}
