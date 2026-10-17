//! An explicit build preset. Profiling features never leak into game artifacts.
use std::path::Path;

use crate::{
    authoring, pipeline,
    process::{NPM, WASM_TARGET, clients, run},
};

/// With `attribution`, hot engine helpers stay out of line so sampling
/// profilers can separate their costs. That build has its own output
/// directories: its timings are not those of the ordinary profiling build.
pub fn build(root: &Path, attribution: bool) -> Result<(), String> {
    let (features, name, mode) = if attribution {
        (
            "engine/profiling,engine/dev,engine/attribution",
            "attribution",
            "attribution",
        )
    } else {
        ("engine/profiling,engine/dev", "profile", "profiling")
    };
    authoring::validate_sources(root)?;
    // Produce ordinary bindings independently before Cargo feature unification
    // can enable instrumentation in any shared dependency.
    pipeline::build_wasm(root, false, "release")?;
    // Every game; job workers keep their ordinary bindings.
    let games: Vec<_> = clients()
        .iter()
        .filter(|client| client.game.is_some())
        .collect();
    let mut arguments = Vec::new();
    for game in &games {
        arguments.extend(["-p", game.package.as_str()]);
    }
    arguments.extend([
        "--features",
        features,
        "--target",
        WASM_TARGET,
        "--profile",
        "profiling",
        "--locked",
    ]);
    let build = xtask::builds::cargo(root, &arguments, "browser-profile")?;
    for game in &games {
        let artifact = build.artifact(&game.wasm, "wasm")?;
        xtask::bindings::generate(
            root,
            &artifact,
            &game.wasm,
            &["--keep-debug"],
            &root.join(format!("web/pkg-{name}")),
        )?;
    }
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
