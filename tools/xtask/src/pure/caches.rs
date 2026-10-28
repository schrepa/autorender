use std::path::Path;
use std::process::Command;

use serde_json::Value;

use super::Observations;

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
