//! Shared native entry point; the Node runner owns browser/report lifetimes.

use std::{path::Path, process::Command};

use serde_json::json;

pub fn benchmark(root: &Path, arguments: &[String]) -> Result<(), String> {
    if let Err(error) = crate::pipeline::install_web_dependencies(root) {
        println!(
            "{}",
            json!({"version":1,"ok":false,"error":{"code":"benchmark_setup_failed","path":"web/node_modules","message":error}})
        );
        return Err("benchmark setup failed; see its JSON result".into());
    }
    let status = Command::new("node")
        .arg("web/scripts/benchmark.mjs")
        .args(arguments)
        .current_dir(root)
        .status();
    match status {
        Ok(status) if status.success() => Ok(()),
        Ok(_) => Err("benchmark failed; see its JSON result".into()),
        Err(error) => {
            println!(
                "{}",
                json!({"version":1,"ok":false,"error":{"code":"benchmark_runner_failed","path":"node","message":error.to_string()}})
            );
            Err("cannot launch benchmark runner; see its JSON result".into())
        }
    }
}
