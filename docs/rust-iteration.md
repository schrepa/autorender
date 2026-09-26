# Automatic Rust iteration

Start the complete development loop with one command:

```powershell
. ./scripts/activate.ps1
cargo xtask agent watch
```

This installs locked web dependencies when needed, starts a managed web server at `http://127.0.0.1:5174/playground.html`, starts the agent bridge on port 7331, builds development Wasm, and launches a paused Chrome session. It then watches Rust/build inputs. Use `--web-port`, `--port`, `--game orbit`, or `--browser PATH` to select another configuration. An existing ordinary server on port 5173 can remain running.

The first build uses a separate Cargo cache at `target/iteration-wasm`; later edits reuse unchanged Cargo build artifacts. Builds currently use the release profile. The supervisor is one foreground process; Ctrl+C or the identified `close` command stops its browsers, owned compiler process tree, web server, and bridge.

## Observe and edit

Use the Node client while editing Rust, including when shared engine code does not compile:

```sh
node web/scripts/agent.mjs development --port 7331
node web/scripts/agent.mjs sessions --port 7331
node web/scripts/agent.mjs inspect --port 7331 --session SESSION_ID --json '{"entity":"player","component":"Transform"}'
```

The direct client does not compile Rust. `cargo xtask agent ...` is a convenience launcher; Cargo can fail before launching it when shared Rust is broken, and Windows can lock a native executable while it runs. The watcher keeps using its already compiled build worker. Changes to native tooling require stopping the watcher, rebuilding it, and starting again.

`development` returns the active build/session separately from the latest attempt. Each attempt has a build ID and source fingerprint. Compiler failures return bounded `diagnostics` with level, code, message, source file/line/column spans, and help entries, plus truncation counters. Manifest, toolchain, binding-generation, and browser-startup failures use structured error codes. Full results are queried over HTTP; stdout emits compact progress events.

Edit a `.rs` file and save it. The watcher detects the content change, debounces saves, and queues at most the newest pending version. Only one build runs at a time. If inputs change while compiling or starting a browser, the obsolete candidate is discarded. Failed compilation or browser startup preserves the previous game's state and session. Fixing the source triggers another build automatically.

After a successful build, the supervisor prepares a new paused browser using that build's immutable bindings. It waits for the development API and an initial render, drains accepted old-session commands, and switches to a fresh session ID. New arrivals during that brief boundary receive `session_replacing`. Old IDs then receive `session_mismatch`; commands are never replayed. Read `development` or `sessions` again and use the new session for inspection and capture.

This is a fresh game instance: gameplay state and temporary input are reset. Saved scene edits remain on disk and are loaded by the new game. Scene-only edits still use the existing validate/save/reload commands and do not trigger Rust builds.

## Ownership and limits

The source fingerprint covers files under `crates`, `examples`, `.cargo`, workspace Cargo manifests/lockfile, the pinned toolchain, and the embedded generated scene schema. It detects additions, deletions, and content edits without trusting timestamps. Generated directories, web scene sources, host TypeScript, and native tool sources are excluded. Host/tooling changes require restarting this managed workflow; ordinary `cargo xtask dev` remains available for Vite host editing.

Polling occurs outside the browser and costs one bounded source scan approximately twice per second. Input limits are 20,000 files and 128 MiB. Symlinked inputs are rejected explicitly. Compiler diagnostics retain at most 100 entries and 256 KiB of diagnostic JSON. The supervisor retains a bounded latest result, one pending fingerprint, three successful binding generations, and at most two browsers during replacement. Failed/superseded generations are removed. A worker has a ten-minute deadline; shutdown stops only its owned process tree.

Bindings live in ignored `web/pkg-iterations/BUILD_ID`, separate from production `web/pkg` and ordinary development `web/pkg-dev`. Running pages are pinned to a complete generation. Open the base game URL for the latest active build; old pinned URLs expire after generation retirement. Production builds do not expose generation selection or the development interface.

Only one watcher owns a working copy, enforced by `web/pkg-iterations/.watch.lock`. Other working copies use independent source/output directories and ports. After a crash, confirm the recorded process has stopped before removing a stale lock. Startup does not automatically delete directories left by an earlier crashed process.

## Verification

`cargo xtask test` covers content watching, coalescing, failed builds, superseded activation, queue boundaries, structured output bounds, and shutdown decisions. With local Rust tools and Chrome available:

```sh
node web/scripts/iteration-smoke.mjs
```

The smoke temporarily edits the sample movement source, creates a compiler error, verifies the old game remains inspectable, fixes it with a changed movement speed, and verifies the new behavior and session in Chrome. It saves original bytes under `artifacts/agent`, restores only the bytes it owns, and closes its services. Stop any existing watcher for that working copy before running it.
