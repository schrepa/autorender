//! An explicit build preset. Profiling features never leak into game artifacts.
use std::path::Path;

use crate::{
    authoring, pipeline,
    process::{NPM, WASM_TARGET, run},
};

/// With `attribution`, hot engine helpers stay out of line so sampling
/// profilers can separate their costs. That build has its own output
/// directories: its timings are not those of the ordinary profiling build.
pub fn build(root: &Path, attribution: bool) -> Result<(), String> {
    let (features, name, mode) = if attribution {
        ("profiling,dev,attribution", "attribution", "attribution")
    } else {
        ("profiling,dev", "profile", "profiling")
    };
    authoring::validate_sources(root)?;
    // Produce ordinary bindings independently before Cargo feature unification
    // can enable instrumentation in any shared dependency.
    pipeline::build_wasm(root, false)?;
    run(
        root,
        "cargo",
        &[
            "build",
            "-p",
            "stress-lab",
            "--features",
            features,
            "--target",
            WASM_TARGET,
            "--profile",
            "profiling",
            "--locked",
        ],
    )?;
    run(
        root,
        "wasm-bindgen",
        &[
            &format!("target/{WASM_TARGET}/profiling/stress_lab.wasm"),
            "--target",
            "web",
            "--out-dir",
            &format!("web/pkg-{name}"),
            "--out-name",
            "stress_lab",
            "--keep-debug",
        ],
    )?;
    pipeline::install_web_dependencies(root)?;
    run(
        root,
        NPM,
        &["run", "build", "--prefix", "web", "--", "--mode", mode],
    )?;
    let (label, port) = if attribution {
        ("Attribution", 5176)
    } else {
        ("Profiling", 5175)
    };
    println!(
        "{label} build: web/dist-{name}. Serve with npm run preview --prefix web -- --outDir dist-{name} --port {port} --strictPort"
    );
    Ok(())
}
