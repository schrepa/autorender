//! Discoverable developer commands, available without compiling the engine.

/// Every command: its usage and what it does, as `help` prints them and
/// `help --json` lists them.
fn commands() -> Vec<(String, &'static str)> {
    let games = "NAME";
    vec![
        ("cache status | prune | configure --max-bytes N".into(), "Inspect or bound disposable preparation shared by worktrees. Active OS leases prevent eviction. ENGINE_CACHE_DIR selects a cache parent; ENGINE_CACHE_MAX_BYTES overrides the persisted idle logical-byte budget. A hard active-build limit requires an OS quota. Caller reports and foreign targets are untouched."),
        ("prepare helper | prepare-native --package NAME [--profile PROFILE] [--features LIST]".into(), "Prepare verified tools or a native bundle. Cargo validates real build inputs. Helpers are copied before execution so Windows can rebuild while an earlier command runs."),
        ("prepare-binaries [--hold] -- CARGO_BUILD_ARGUMENTS".into(), "Build selected binaries into the shared context and return immutable paths. Changing workspaces clears local-package outputs through Cargo while retaining registry dependencies. --hold pins bundles until the caller closes stdin, for scripts that run several tools."),
        ("run-native --package NAME [--profile PROFILE] [--features LIST] -- ARGS...".into(), "Run a fresh native program while holding its immutable bundle lease; inherits streams and process group."),
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
        ("smoke --list | [--serve] [--keep-going] [--native-profile PROFILE] (--all | NAME...)".into(), "List or run the browser smoke checks: against the development server DEMO_URL names, or with --serve one of this checkout started on a free port (never 5173-5175) and stopped after; holding the machine's heavy-run lock; stopping at the first failure, or with --keep-going running every one and listing each result; --native-profile builds the native programs checks run with that cargo profile (dev: debug assertions on); exit 3 when an assertion failed"),
        ("pure [--bless] [--wasm] [CORPUS...]".into(), "Run the deterministic checks of engine code over tracked fixtures, without a GPU or browser (corpora: terrain, scenes, simplify, gltf, import, caches): each checks what the code promises and that its outputs equal the goldens in tools/xtask/pure/goldens.json; --bless rewrites them from a run whose promises hold, for review; --wasm replays the terrain corpus's jobs in the browser worker's Wasm module with Node (web/pkg-dev, built first) and requires the native bits; exit 3 when a promise broke or an output changed"),
        ("core-smoke [--bless]".into(), "Run the core smoke (the caches corpus of pure): tools/coresmoke in the dev profile, debug assertions on, checks the incremental caches against fresh engines under seeded random edits, and holds a deterministic terrain streaming trace (written to target/coresmoke/terrain-trace.txt) and the Stress Lab's content digests to their goldens; a failure prints the seed and the edits"),
        ("gpu".into(), "Check the renderer's shaders against its Rust GPU layouts and bindings"),
        ("gpu --report DIR [--no-rga] [--no-cache]".into(), "Report every shader entry point's cost offline, on any machine: naga's counts and SPIR-V, and with Radeon GPU Analyzer (downloaded once, Windows and Linux) its AMD machine code, registers and memory on RDNA 3, 3.5 and 4"),
        ("bench check".into(), "Check bench/ against the code: the metric catalog names what measurements report, the scene and settings files known options, the shader paths every shader"),
        ("bench assets".into(), "Build the files workloads read in the shared bounded cache (keyed by actual generator binaries and arguments) and write them into web/public/stress for the page; prints the folder and its generated_hash"),
        ("measure --request REQUEST.json --out REPORT.json [--extend REPORT.json] | --schema".into(), "Run a local paired native comparison with explicit revisions, workload arms, method, setup and environment. Fresh processes alternate AB/BA. Preserve raw attempts and checkpoints; continuation requires compatible inputs and the same actual tool and artifacts. Reports use relative artifact manifests."),
        ("equivalence [--base REV] [--candidate REV] [--tier smoke|full] [--cells NAME,...] [--aa] [--repeat N] [--gpu discrete|integrated] [--browser | --both [--hosts]]".into(), "Whether a change leaves what the engine draws and does as it was: the base (the merge-base with origin/main, or --base) and the candidate (the working tree, or --candidate) each build the Stress Lab, then every cell of bench/equivalence.json's tier runs on both, settled and captured, natively (default), in headless Chrome against each side's development server (--browser) or both (--both; --hosts measures how the candidate's native and Chrome images differ); captures compare by hash then pixel, judged work counters by their rule, digests by part; an unstable capture (its reason in the matrix) passes when it matches one of the base's --repeat runs or lies no farther from them than they lie from each other; --aa runs the base against itself; each cell holds the heavy-run lock; exit 0 equal, 3 different, 1 could not check"),
        ("checks --request REQUEST.json --out REPORT.json | --schema".into(), "Run revision-specific browser checks with independent script revisions, bounded subprocesses and every attempt retained. No external service or task record required."),
        ("selfcheck [measure] [checks] | tooling".into(), "Check the execution tools against themselves on the working tree as a commit (made without moving any ref): an A/A measure task (both arms that commit, Field, 2 clean pairs) must complete with every run settled, the same work counted every run and the arms' median frame times within 25%; one check run with its commit's own scripts and with the same commit's scripts checked out again must hold at its first attempt both ways, and expected to fail must fail every attempt; files in target/selfcheck/; exit 3 when the sides differed"),
        ("verify [--level quick|pr|merge] [--base REV] [--plan]".into(), "Run the checks a change calls for: the paths the working tree changed since the base (the merge-base with origin/main, or --base) select them through the path map (tools/xtask/src/verify/map.rs); quick (under 2 minutes) runs formatting, the drift checks, the shaders, Clippy over the changed packages and the selected pure corpora; pr (the default, under 10 minutes) all of lint, the selected smokes with --serve, the Wasm replay where selected, equivalence's smoke tier plus the selected cells natively and the selected self-checks of the execution tools; merge (30-45 minutes) all of lint, every smoke, the native smoke again built in the dev profile (debug assertions on), the Wasm replay, equivalence's full tier and, where selected, its browser leg and the self-checks; --plan prints the steps without running them; exit 3 when a step found a fault, 1 when one could not check"),
        ("lint".into(), "Run every GPU-free check, cheapest first and each whatever failed before it: the repository policy, formatting, the web host's text checks, scene sources, generated files, bench/, smoke coverage, feature layering, the schedules' ordering of conflicting systems, shaders against Rust, the pure checks, native and Wasm Clippy, rustdoc, the browser clients and TypeScript; ends with each stage's result and time"),
        ("docs".into(), "Generate the documentation of every item, private ones included, of the native workspace and the browser façade, in the shared validation context printed by Cargo; fails where a link does not resolve, public docs link a private item, or markup is malformed, every rustdoc lint denied (lint's docs stage)"),
        ("help [--json]".into(), "Show this message, or the commands as JSON"),
    ]
}

pub fn help(options: &[String]) -> Result<(), String> {
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
