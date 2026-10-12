//! `cargo xtask lint`: every check that needs no GPU, as a table of stages
//! run cheapest first. Every stage runs whatever failed before it, unless a
//! stage it needs did not pass; the run ends with each stage's result and
//! time, and fails if any stage did not pass.

use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use crate::{
    authoring,
    pipeline::{QUICK, build_wasm, install_web_dependencies},
    process::{DEV_FEATURES, NPM, WASM_TARGET, client_packages, run},
    structure,
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
/// generated files, bench/ and its equivalence matrix, smoke coverage, each
/// smoke's opening description, the path map `cargo xtask verify` selects
/// checks by, feature layering, the
/// facade's exports), the structure ratchet (file
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
        run: fmt,
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
        name: "smoke descriptions",
        needs: &[],
        run: crate::smoke::check,
    },
    Stage {
        name: "path map",
        needs: &[],
        run: crate::verify::check,
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

/// Paths one rustfmt command line takes: a few hundred stay well within
/// the longest command line Windows accepts.
const RUSTFMT_BATCH: usize = 200;

/// Formatting: cargo fmt's check over every crate's module tree, then
/// rustfmt's over every Rust source git would commit, which reaches the
/// modules a macro declares too (rustfmt expands no macros, so cargo fmt
/// never finds them). Vendored and generated sources keep their tools'
/// layout.
fn fmt(root: &Path) -> Result<(), String> {
    run(root, "cargo", &["fmt", "--all", "--", "--check"])?;
    let edition = edition(root)?;
    let files: Vec<String> = structure::sources(root)?
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
                // Windows' verbatim prefix, which rustfmt's paths carry.
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
            "rustfmt finds all {} Rust sources formatted, macro-declared modules included.",
            files.len()
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

/// The Rust edition of the workspace's `[workspace.package]`, which every
/// crate inherits.
fn edition(root: &Path) -> Result<String, String> {
    let manifest = fs::read_to_string(root.join("Cargo.toml"))
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
