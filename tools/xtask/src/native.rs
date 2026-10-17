//! Native developer launches keep immutable program bundles leased for the
//! entire child lifetime. A concurrent build can replace Cargo outputs safely.

use crate::{builds, cache::Entry};
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    process::Command,
};

fn prepare(root: &Path, args: &[String]) -> Result<(Entry, String, usize), String> {
    let mut package = None;
    let mut profile = "release";
    let mut features = "dev";
    let mut index = 0;
    while index < args.len() && args[index] != "--" {
        let value = args
            .get(index + 1)
            .ok_or("native option requires a value")?;
        match args[index].as_str() {
            "--package" => package = Some(value.as_str()),
            "--profile" => profile = value,
            "--features" => features = value,
            _ => return Err(format!("unknown native option {}", args[index])),
        }
        index += 2;
    }
    let package = package.ok_or("--package is required")?;
    let name = format!("{package}-native");
    let build = builds::native_binary(
        root,
        &[
            "-p",
            package,
            "--bin",
            &name,
            "--profile",
            profile,
            "--features",
            features,
            "--locked",
        ],
        "native",
        &[],
    )?;
    let libraries: Vec<PathBuf> = std::env::var_os("DXC_PATH")
        .map(PathBuf::from)
        .into_iter()
        .collect();
    let bundle = builds::bundle(root, &build, &name, &libraries)?;
    Ok((bundle, name, index.saturating_add(1)))
}

/// Build-only preparation returns provenance; run holds a lease and inherits
/// the caller's process group and streams so normal supervisor cancellation
/// reaches the whole tree and the engine protocol remains unchanged.
pub fn command(root: &Path, args: &[String], run: bool) -> Result<u8, String> {
    let (bundle, name, after) = prepare(root, args)?;
    let executable = bundle
        .data()
        .join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
    if !run {
        println!(
            "{}",
            json!({"version": 1, "program": executable, "manifest": bundle.manifest()?})
        );
        return Ok(0);
    }
    let status = Command::new(executable)
        .args(args.get(after..).unwrap_or_default())
        .current_dir(root)
        .status()
        .map_err(|e| e.to_string())?;
    Ok(status
        .code()
        .and_then(|code| u8::try_from(code).ok())
        .unwrap_or(1))
}
