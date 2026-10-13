//! The caches corpus: the core smoke (tools/coresmoke), built in the dev
//! profile so the engine's debug assertions run, over the generated meshes
//! the dense scene reads (built as `cargo xtask bench assets` builds them,
//! rebuilt only when their generator changes). Its incremental caches must
//! agree with fresh engines', and its
//! terrain streaming trace and Stress Lab content digests are observations
//! held to goldens. `cargo xtask core-smoke` runs it alone.

use std::path::Path;
use std::process::Command;

use serde_json::Value;

use super::Observations;

/// Where the smoke writes its terrain streaming trace, one line a frame,
/// for a diff when the trace's hash changes.
pub const TRACE: &str = "target/coresmoke/terrain-trace.txt";

pub fn run(root: &Path, observed: &mut Observations) -> Result<(), String> {
    let folder = crate::bench::dense_assets(root)?.display().to_string();
    let trace = root.join(TRACE).display().to_string();
    println!("Running the core smoke (cargo run -p coresmoke)");
    let output = Command::new("cargo")
        .args(["run", "-q", "--locked", "-p", "coresmoke", "--target-dir"])
        .arg(crate::process::tools_target(root))
        .arg("--")
        .args(["--assets", &folder, "--trace", &trace])
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot run cargo: {error}"))?;
    // 3: it ran and found a cache disagreeing; anything else but success
    // stopped it.
    if !(output.status.success() || output.status.code() == Some(3)) {
        return Err(format!(
            "the core smoke stopped ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let report: Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("the core smoke printed no report: {error}"))?;
    for (name, value) in report["observations"].as_object().into_iter().flatten() {
        observed.record(name.clone(), value.clone());
    }
    for broken in report["broken"].as_array().into_iter().flatten() {
        let broken = broken.as_str().unwrap_or_default().to_owned();
        observed.check(false, || broken);
    }
    Ok(())
}
