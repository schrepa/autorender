//! The repository's tasks: building, serving, checking and measuring the
//! engine and its games (`cargo xtask help` lists them).

mod ambiguities;
mod authoring;
mod bench;
mod checkouts;
mod checks;
mod coverage;
mod development;
mod equivalence;
mod exports;
mod features;
mod formats;
mod gpu;
mod heavy;
mod importing;
mod iteration_build;
mod lint;
mod locality;
mod measure;
mod pipeline;
mod process;
mod profiling;
mod pure;
mod scaffold;
mod selfcheck;
mod serve;
mod shader_report;
mod smoke;
mod structure;
mod verify;

use std::{env, process::ExitCode};

/// How a command failed, and the exit code it ends with: 1 when it could
/// not do what it was asked, 3 when it did and found what it checks wrong
/// (a smoke's failed assertion), as the smokes' own exit codes say.
pub struct Failure {
    pub message: String,
    pub code: u8,
}

impl Failure {
    /// A check that ran and found what it checks wrong: exit code 3.
    pub fn found(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: 3,
        }
    }
}

impl From<String> for Failure {
    fn from(message: String) -> Self {
        Self { message, code: 1 }
    }
}

impl From<&str> for Failure {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            eprintln!("xtask: {}", failure.message);
            ExitCode::from(failure.code)
        }
    }
}

fn execute() -> Result<(), Failure> {
    let mut arguments = env::args().skip(1);
    let task = arguments.next().unwrap_or_else(|| "help".to_owned());
    let options: Vec<_> = arguments.collect();
    let root = process::repository_root()?;
    if task == "dev" {
        return Ok(pipeline::dev(&root, dev_port(&options)?)?);
    }
    if task == "agent" {
        return Ok(development::agent(&root, &options)?);
    }
    if task == "scene" {
        return Ok(development::scene(&options)?);
    }
    if task == "asset" {
        return Ok(importing::asset(&options)?);
    }
    if task == "profile" {
        return Ok(match options.as_slice() {
            [] => profiling::build(&root, false),
            [option] if option == "--attribution" => profiling::build(&root, true),
            _ => Err("usage: profile [--attribution]".into()),
        }?);
    }
    if task == "smoke" {
        return smoke::smoke(&root, &options);
    }
    if task == "pure" {
        return pure::pure(&root, &options);
    }
    if task == "core-smoke" {
        let mut corpus = vec!["caches".to_owned()];
        corpus.extend(options);
        return pure::pure(&root, &corpus);
    }
    if task == "bench" {
        return Ok(bench::bench(&root, &options)?);
    }
    if task == "measure" {
        return Ok(measure::measure(&root, &options)?);
    }
    if task == "equivalence" {
        return equivalence::equivalence(&root, &options);
    }
    if task == "selfcheck" {
        return selfcheck::selfcheck(&root, &options);
    }
    if task == "verify" {
        return verify::verify(&root, &options);
    }
    if task == "checks" {
        return Ok(checks::check_task(&root, &options)?);
    }
    if task == "new" {
        return Ok(scaffold::new(&root, &options)?);
    }
    if task == "gpu" && !options.is_empty() {
        return Ok(shader_report::run(&root, &options)?);
    }
    if task == "build-dev" {
        let [option, output] = options.as_slice() else {
            return Err("usage: build-dev --output web/pkg-iterations/GENERATION".into());
        };
        if option != "--output" {
            return Err("build-dev requires --output".into());
        }
        let result = iteration_build::build(&root, std::path::Path::new(output));
        println!("{result}");
        return if result["ok"] == true {
            Ok(())
        } else {
            Err("development build failed; see JSON diagnostics".into())
        };
    }
    if matches!(task.as_str(), "help" | "--help" | "-h") {
        return Ok(help(&options)?);
    }
    if let Some(argument) = options.first() {
        return Err(
            format!("unexpected argument {argument:?}; run `cargo xtask help` for usage").into(),
        );
    }
    Ok(match task.as_str() {
        "build" => pipeline::build(&root),
        "lint" => lint::run_stages(&root, lint::LINT),
        "docs" => pipeline::docs(&root),
        "assets" => authoring::validate_sources(&root),
        "schema" => authoring::write_generated(&root),
        "gpu" => gpu::check(&root),
        _ => Err(format!(
            "unknown command {task:?}; run `cargo xtask help` for usage"
        )),
    }?)
}

fn dev_port(arguments: &[String]) -> Result<u16, String> {
    match arguments {
        [] => Ok(5173),
        [option, value] if option == "--port" => value
            .parse::<u16>()
            .ok()
            .filter(|port| *port != 0)
            .ok_or_else(|| "dev --port requires an integer from 1 to 65535".to_owned()),
        _ => Err("usage: cargo xtask dev [--port N]".to_owned()),
    }
}

/// Every command: its usage and what it does, as `help` prints them and
/// `help --json` lists them.
fn commands() -> Vec<(String, &'static str)> {
    let games = process::game_names();
    vec![
        ("build".into(), "Build all release Wasm clients and bundle the web host"),
        ("profile [--attribution]".into(), "Build isolated instrumented games in web/dist-profile; --attribution keeps hot helpers out of line for sampling profilers, in web/dist-attribution"),
        ("dev [--port N]".into(), "Build and serve on loopback (default port 5173)"),
        (format!("agent watch [--web-port N] [--port N] [--game {games}]"), "Watch Rust, build, and replace identified browser sessions"),
        ("agent help".into(), "The agent client's verbs and options and the engine's queries and controls, as JSON"),
        ("agent VERB [--session S] [--json JSON | --file PATH]".into(), "Run a development query or control in a browser session"),
        ("scene validate --file PATH | save --file TARGET --candidate PATH --expected-revision N | schema".into(), "Validate or save revision-checked scene sources, or print the scene schema, as JSON"),
        ("asset import --file PATH --id ID (--output PATH | --into PATH --expected-revision N) [--replace] [--as-prefab] [--virtual-triangles N | --no-virtual-geometry]".into(), "Import a static textured GLB with structured results; dense opaque primitives (512 triangles and up by default) become virtual meshes"),
        ("assets".into(), "Validate authored scene sources without changing them"),
        ("schema".into(), "Write the generated protocol summary, client list, stats and game surface types, and each app's own"),
        ("new app NAME".into(), "Start a game in apps/NAME with its page web/NAME.html, from the engine's app template"),
        ("smoke --list | [--serve] [--keep-going] (--all | NAME...)".into(), "List or run the browser smoke checks: against the development server DEMO_URL names, or with --serve one of this checkout started on a free port (never 5173-5175) and stopped after; holding the machine's heavy-run lock; stopping at the first failure, or with --keep-going running every one and listing each result; exit 3 when an assertion failed"),
        ("pure [--bless] [--wasm] [CORPUS...]".into(), "Run the deterministic checks of engine code over tracked fixtures, without a GPU or browser (corpora: terrain, scenes, simplify, gltf, import, caches): each checks what the code promises and that its outputs equal the goldens in tools/xtask/pure/goldens.json; --bless rewrites them from a run whose promises hold, for review; --wasm replays the terrain corpus's jobs in the browser worker's Wasm module with Node (web/pkg-dev, built first) and requires the native bits; exit 3 when a promise broke or an output changed"),
        ("core-smoke [--bless]".into(), "Run the core smoke (the caches corpus of pure): tools/coresmoke in the dev profile, debug assertions on, checks the incremental caches against fresh engines under seeded random edits, and holds a deterministic terrain streaming trace (written to target/coresmoke/terrain-trace.txt) and the Stress Lab's content digests to their goldens; a failure prints the seed and the edits"),
        ("gpu".into(), "Check the renderer's shaders against its Rust GPU layouts and bindings"),
        ("gpu --report DIR [--no-rga]".into(), "Report every shader entry point's cost offline, on any machine: naga's counts and SPIR-V, and with Radeon GPU Analyzer (downloaded once, Windows and Linux) its AMD machine code, registers and memory on RDNA 2, 3 and 4"),
        ("bench check".into(), "Check bench/ against the code: the metric catalog names what measurements report, the scene and settings files known options, the shader paths every shader"),
        ("bench assets".into(), "Build the files workloads read that git does not hold into target/bench-assets/<inputs hash>/ (reused while the generators are unchanged) and write them into web/public/stress for the page; prints the folder and its generated_hash"),
        ("measure --task TASK.json --profile PROFILE.json --out REPORT.json [--extend REPORT.json] [--formats FORMATS.json]".into(), "Run an engine-factory measure task on the native host from this checkout (the task's tools commit): build each commit it names, run each workload's two arms (each its commit's program from that commit's tree, with its own switches, scene, settings and generated files) in alternating rounds until its clean pairs, profile each arm and report the shaders where the task wants them, run the task's checks, and write the factory's measure report, checked against FORMATS (the factory's /api/formats) when given"),
        ("equivalence [--base REV] [--candidate REV] [--tier smoke|full] [--cells NAME,...] [--aa] [--repeat N] [--keep N] [--gpu discrete|integrated] [--browser | --both [--hosts]]".into(), "Whether a change leaves what the engine draws and does as it was: the base (the merge-base with origin/main, or --base) and the candidate (the working tree, or --candidate) each build the Stress Lab, then every cell of bench/equivalence.json's tier runs on both, settled and captured, natively (default), in headless Chrome against each side's development server (--browser) or both (--both; --hosts measures how the candidate's native and Chrome images differ); captures compare by hash then pixel, judged work counters by their rule, digests by part; an unstable capture (its reason in the matrix) passes when it matches one of the base's --repeat runs or lies no farther from them than they lie from each other; --aa runs the base against itself; each cell holds the heavy-run lock; exit 0 equal, 3 different, 1 could not check"),
        ("checks --task TASK.json --profile PROFILE.json --out REPORT.json [--formats FORMATS.json]".into(), "Run an engine-factory check task from this checkout: each check's script against a development server of its commit, every attempt kept (a failing check runs again, up to 3 attempts), as the factory's check report, checked against FORMATS when given"),
        ("selfcheck [measure] [checks]".into(), "Check the factory's tools against themselves on the working tree as a commit (made without moving any ref): an A/A measure task (both arms that commit, Field, 2 clean pairs) must complete with every run settled, the same work counted every run and the arms' median frame times within 25%; one check run with its commit's own scripts and with the same commit's scripts checked out again must hold at its first attempt both ways, and expected to fail must fail every attempt; files in target/selfcheck/; exit 3 when the sides differed"),
        ("verify [--level quick|pr|merge] [--base REV] [--plan]".into(), "Run the checks a change calls for: the paths the working tree changed since the base (the merge-base with origin/main, or --base) select them through the path map (tools/xtask/src/verify/map.rs); quick (under 2 minutes) runs formatting, the drift checks, the shaders, Clippy over the changed packages and the selected pure corpora; pr (the default, under 10 minutes) all of lint, the selected smokes with --serve, the Wasm replay where selected, equivalence's smoke tier plus the selected cells natively and the selected self-checks of the factory's tools; merge (30-45 minutes) all of lint, every smoke, the Wasm replay, equivalence's full tier and, where selected, its browser leg and the self-checks; --plan prints the steps without running them; exit 3 when a step found a fault, 1 when one could not check"),
        ("lint".into(), "Run every GPU-free check, cheapest first and each whatever failed before it: the repository policy, formatting, the web host's text checks, scene sources, generated files, bench/, smoke coverage, feature layering, the schedules' ordering of conflicting systems, shaders against Rust, the pure checks, native and Wasm Clippy, rustdoc, the browser clients and TypeScript; ends with each stage's result and time"),
        ("docs".into(), "Generate the documentation of every item, private ones included, of the native workspace and the browser façade, in target/doc and target/wasm32-unknown-unknown/doc; fails where a link does not resolve, public docs link a private item, or markup is malformed, every rustdoc lint denied (lint's docs stage)"),
        ("help [--json]".into(), "Show this message, or the commands as JSON"),
    ]
}

fn help(options: &[String]) -> Result<(), String> {
    match options {
        [] => {
            println!(
                "Usage: cargo xtask <command>

Commands:"
            );
            for (usage, summary) in commands() {
                println!(
                    "  {usage}
      {summary}"
                );
            }
            Ok(())
        }
        [json] if json == "--json" => {
            let commands: Vec<_> = commands()
                .into_iter()
                .map(|(usage, summary)| serde_json::json!({"usage": usage, "summary": summary}))
                .collect();
            println!("{}", serde_json::json!({ "commands": commands }));
            Ok(())
        }
        _ => Err("usage: cargo xtask help [--json]".into()),
    }
}
