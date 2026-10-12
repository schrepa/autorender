//! `cargo xtask lint`: every check that needs no GPU, as a table of stages
//! run cheapest first. Every stage runs whatever failed before it, unless a
//! stage it needs did not pass; the run ends with each stage's result and
//! time, and fails if any stage did not pass.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::{
    authoring,
    pipeline::{QUICK, build_wasm, install_web_dependencies},
    process::{DEV_FEATURES, NPM, WASM_TARGET, client_packages, run},
};

/// One step of a run: its name, the stages that must have passed for it to
/// run (it is skipped otherwise), and what it does.
pub(crate) struct Stage {
    pub name: &'static str,
    pub needs: &'static [&'static str],
    pub run: fn(&Path) -> Result<(), String>,
}

/// What `cargo xtask lint` runs, in order: the repository policy,
/// formatting, the web host's text checks, the drift checks (scene sources,
/// generated files, bench/ and its equivalence matrix, smoke coverage,
/// feature layering, the facade's exports), the structure ratchet (file
/// lengths, struct widths), shaders against their Rust layouts, the pure
/// checks' corpora, Clippy (the workspace, the native host's tools, the
/// browser clients, the app template), rustdoc, the browser clients the web
/// host imports, and TypeScript.
pub(crate) const LINT: &[Stage] = &[
    Stage {
        name: "policy",
        needs: &[],
        run: policy::check,
    },
    Stage {
        name: "fmt",
        needs: &[],
        run: |root| run(root, "cargo", &["fmt", "--all", "--", "--check"]),
    },
    Stage {
        name: "web layers",
        needs: &[],
        run: |root| web_check(root, "check-layers.mjs"),
    },
    Stage {
        name: "web errors",
        needs: &[],
        run: |root| web_check(root, "check-errors.mjs"),
    },
    Stage {
        name: "web parameters",
        needs: &[],
        run: |root| web_check(root, "check-parameters.mjs"),
    },
    Stage {
        name: "scene sources",
        needs: &[],
        run: authoring::validate_sources,
    },
    Stage {
        name: "generated files",
        needs: &[],
        run: authoring::check_generated,
    },
    Stage {
        name: "bench",
        needs: &[],
        run: crate::bench::check,
    },
    Stage {
        name: "equivalence matrix",
        needs: &[],
        run: crate::equivalence::check,
    },
    Stage {
        name: "coverage",
        needs: &[],
        run: crate::coverage::check,
    },
    Stage {
        name: "features",
        needs: &[],
        run: crate::features::check,
    },
    Stage {
        name: "exports",
        needs: &[],
        run: crate::exports::check,
    },
    Stage {
        name: "structure",
        needs: &[],
        run: crate::structure::check,
    },
    Stage {
        name: "gpu",
        needs: &[],
        run: crate::gpu::check,
    },
    Stage {
        name: "pure",
        needs: &[],
        run: |root| crate::pure::pure(root, &[]).map_err(|failure| failure.message),
    },
    Stage {
        name: "clippy",
        needs: &[],
        run: |root| clippy(root, &["--workspace", "--all-targets"]),
    },
    // The native host with its development tools, and with its profiler,
    // as agents build native games (the app template's entry included).
    Stage {
        name: "clippy native dev",
        needs: &[],
        run: |root| clippy_native(root, "engine_native/dev,stress-lab/dev"),
    },
    Stage {
        name: "clippy native profiling",
        needs: &[],
        run: |root| {
            clippy_native(
                root,
                "engine_native/dev,engine_native/profiling,stress-lab/dev,stress-lab/profiling",
            )
        },
    },
    // The clients for the browser, without and with their development
    // tools.
    Stage {
        name: "clippy wasm",
        needs: &[],
        run: |root| clippy_wasm(root, false),
    },
    Stage {
        name: "clippy wasm dev",
        needs: &[],
        run: |root| clippy_wasm(root, true),
    },
    // The app template, as new apps start from it.
    Stage {
        name: "clippy app template",
        needs: &[],
        run: |root| {
            clippy(
                root,
                &[
                    "-p",
                    "engine",
                    "--examples",
                    "--features",
                    "dev",
                    "--target",
                    WASM_TARGET,
                ],
            )
        },
    },
    // Every item's documentation, natively and in the browser façade.
    Stage {
        name: "docs",
        needs: &[],
        run: crate::pipeline::docs,
    },
    // The host imports generated declarations, including on a clean
    // checkout; their types do not depend on the profile. `cargo xtask
    // build` replaces these bindings with release ones before it bundles
    // them.
    Stage {
        name: "wasm",
        needs: &[],
        run: |root| build_wasm(root, false, QUICK),
    },
    Stage {
        name: "wasm dev",
        needs: &[],
        run: |root| build_wasm(root, true, QUICK),
    },
    Stage {
        name: "web dependencies",
        needs: &[],
        run: install_web_dependencies,
    },
    Stage {
        name: "tsc",
        needs: &["wasm", "wasm dev", "web dependencies"],
        run: |root| run(root, NPM, &["run", "typecheck", "--prefix", "web"]),
    },
];

/// Clippy over `selection`, every warning an error.
fn clippy(root: &Path, selection: &[&str]) -> Result<(), String> {
    let mut arguments = vec!["clippy"];
    arguments.extend(selection);
    arguments.extend(["--locked", "--", "-D", "warnings"]);
    run(root, "cargo", &arguments)
}

/// Clippy over the native host, its profiler and the Stress Lab, every
/// target, with `features`.
fn clippy_native(root: &Path, features: &str) -> Result<(), String> {
    clippy(
        root,
        &[
            "-p",
            "engine_native",
            "-p",
            "engine_profiler",
            "-p",
            "stress-lab",
            "--all-targets",
            "--features",
            features,
        ],
    )
}

/// Clippy over the browser clients, with their development tools or
/// without.
fn clippy_wasm(root: &Path, development: bool) -> Result<(), String> {
    let mut selection: Vec<&str> = client_packages().collect();
    if development {
        selection.extend(["--features", DEV_FEATURES]);
    }
    selection.extend(["--target", WASM_TARGET]);
    clippy(root, &selection)
}

/// The web host's text check `script` (in web/scripts), run with Node.
fn web_check(root: &Path, script: &str) -> Result<(), String> {
    run(&root.join("web"), "node", &[&format!("scripts/{script}")])
}

/// How a stage ended.
enum Outcome {
    Passed,
    Failed(String),
    /// Not run: the stages it needs that did not pass.
    Skipped(String),
}

/// Runs `stages` in order, each unless a stage it needs did not pass, then
/// prints each one's result and time; an error naming every stage that
/// failed or was skipped, if any.
pub(crate) fn run_stages(root: &Path, stages: &[Stage]) -> Result<(), String> {
    let mut results: Vec<(&str, Outcome, Duration)> = Vec::with_capacity(stages.len());
    for stage in stages {
        let blocked: Vec<&str> = stage
            .needs
            .iter()
            .copied()
            .filter(|need| {
                !results
                    .iter()
                    .any(|(name, outcome, _)| name == need && matches!(outcome, Outcome::Passed))
            })
            .collect();
        if !blocked.is_empty() {
            results.push((
                stage.name,
                Outcome::Skipped(blocked.join(", ")),
                Duration::ZERO,
            ));
            continue;
        }
        println!("\n== {} ==", stage.name);
        let start = Instant::now();
        let outcome = match (stage.run)(root) {
            Ok(()) => Outcome::Passed,
            Err(error) => {
                eprintln!("{} failed: {error}", stage.name);
                Outcome::Failed(error)
            }
        };
        results.push((stage.name, outcome, start.elapsed()));
    }
    let width = results
        .iter()
        .map(|(name, _, _)| name.len())
        .max()
        .unwrap_or(0)
        .max("stage".len());
    println!("\n{:width$}  result   seconds", "stage");
    for (name, outcome, time) in &results {
        let result = match outcome {
            Outcome::Passed => "ok",
            Outcome::Failed(_) => "FAILED",
            Outcome::Skipped(_) => "skipped",
        };
        println!("{name:width$}  {result:7}  {:7.1}", time.as_secs_f64());
    }
    let total: Duration = results.iter().map(|(_, _, time)| *time).sum();
    println!("{:width$}           {:7.1}", "total", total.as_secs_f64());
    let problems: Vec<String> = results
        .iter()
        .filter_map(|(name, outcome, _)| match outcome {
            Outcome::Passed => None,
            Outcome::Failed(error) => Some(format!("{name} failed: {error}")),
            Outcome::Skipped(blocked) => Some(format!("{name} skipped: needs {blocked}")),
        })
        .collect();
    if problems.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{} of {} stages did not pass:\n{}",
        problems.len(),
        results.len(),
        problems.join("\n")
    ))
}
