# Repository commands

Web dependencies are installed with `npm ci` on a clean checkout or when either
package manifest changes. A stamp inside ignored `node_modules` lets subsequent
commands reuse that exact installation. Stop the workspace's Vite server before
changing dependencies on Windows, where running native modules can be locked.

`xtask` owns repeatable local and automation commands. This host-only Rust
binary depends on the scene authoring API with its optional `schema` feature;
it is never linked into browser games. Run `cargo xtask <command>` from the
repository. Commands locate source files relative to this manifest, inherit the
caller's environment and terminal streams, stop at the first failure, and return
a nonzero exit status. Authoring validation never changes source files.

Prerequisites are the repository's pinned Rust toolchain with the
`wasm32-unknown-unknown` target, Clippy and rustfmt; Node.js 24 or later; npm;
and `wasm-bindgen-cli` **0.2.127** on `PATH`. Install the CLI:

```sh
cargo install wasm-bindgen-cli --version 0.2.127 --locked
```

The CLI version must match the Rust `wasm-bindgen` dependency. Build and
development commands verify it before compiling. Windows launches npm through
`npm.cmd`. Local verification and GitHub Actions use the same commands.

| Command | Behavior |
| --- | --- |
| `cargo xtask build` | Validate scenes, compile all three clients as release Wasm, generate their bindings, run `npm ci`, and bundle the host into `web/dist`. |
| `cargo xtask dev [--port N]` | Validate scenes, compile and bind all three clients, run `npm ci`, then serve on loopback. Port defaults to 5173; valid range is 1–65535. A busy port fails explicitly. |
| `cargo xtask agent ...` | Shared Node/Playwright session CLI: serve, sessions, request, and runtime controls. Results use JSON stdout. |
| `cargo xtask agent watch` | Install locked host dependencies, then own automatic Rust builds and browser session replacement. |
| `cargo xtask agent watch --game stress` | Use the stress lab at `stress.html`; `playground` remains the default and `orbit` selects the original demo. |
| `node web/scripts/agent.mjs development` | Query structured compiler/build status without compiling the CLI. |
| `cargo xtask scene validate --file PATH` | Validate one bounded source and return structured JSON. |
| `cargo xtask scene save --file TARGET --candidate PATH --expected-revision N` | Validate and persist candidate bytes with revision/conflict protection. |
| `cargo xtask assets` | Validate every `web/public/scenes/*.scene.json` in sorted order. Report source path, stable error code and field path on failure; scene ID/revision/entity count on success. |
| `cargo xtask schema` | Generate `docs/schema/scene-v2.schema.json` from Rust authoring types. Review and commit schema changes alongside format changes. |
| `cargo xtask asset import --file MODEL.glb --id NAME --output NEW.scene.json` | Convert the supported static GLB subset into validated scene version 2; return counts and provenance as JSON. |
| `cargo xtask lint` | Validate scenes and schema freshness, then check formatting, native workspace Clippy, all three Wasm clients with Clippy, and TypeScript. Generates bindings for clean checkouts. |
| `cargo xtask test` | Run native workspace tests with development features, then host input, agent transport, and adapter lifecycle contracts through Node's runner. |
| `cargo xtask docs` | Generate native workspace documentation, then Wasm façade/renderer documentation for browser-only APIs. |

`orbit-demo`, `playground` and `stress-lab` build in one Cargo invocation. Their
generated bindings have separate names: `orbit_demo`, `playground` and `stress_lab`.
Production uses `web/pkg`; development enables all clients' `dev` features and uses `web/pkg-dev`.
Lint compiles both presets. This prevents production bundling from overwriting
bindings used by a running development host.
Cargo uses the checked-in lockfile and npm uses `npm ci`. Build commands expect
Cargo's default `target` directory; do not override `CARGO_TARGET_DIR`.

Authoring commands read scene files only after checking the 1 MiB source limit,
then use a bounded reader so concurrent growth cannot bypass that limit. They
call the same parser/validator as browser scene loading. `assets` does not cook
binary assets or save revisions. `lint` compares the generated schema to the
versioned artifact without rewriting it; line-ending differences are ignored.
Schema generation is an explicit operation, with no hidden changes during lint.

Native documentation is under `target/doc`; browser documentation is under
`target/wasm32-unknown-unknown/doc`. Rust edits require rerunning `dev` to rebuild
Wasm; Vite handles host edits. Use distinct `--port` values for independent
working copies. Each copy owns its own source and generated output directories.
For automatic Rust iteration use [agent watch](../../docs/rust-iteration.md),
which keeps a compiled native worker alive through source errors, stages immutable
bindings, and creates new session IDs after successful builds. Its dedicated
Cargo target is `target/iteration-wasm`; native tooling changes require restart.
The watcher fingerprints the entire `examples` tree, including `examples/stress-lab`;
each immutable generation contains all three clients, so switching entry pages
keeps a single complete build identity.
Release Wasm makes the in-game diagnostics useful for the current browser and
machine; it does not establish a general performance guarantee.

The CI workflow runs assets, lint, test, build and docs on Ubuntu using the pinned
Rust and wasm-bindgen versions. It does not assume WebGPU is available in hosted
CI. Run the documented browser smoke checks locally with a supported GPU/browser.

The [agent workflow](../../docs/agent-workflow.md) documents saved edits and
temporary runtime controls. `agent` delegates transport to one shared Node
implementation; `scene` calls `engine_dev::authoring` directly. Both keep stdout
machine-readable and report errors with nonzero status. Native tools never link
into browser games. Use different Vite and bridge ports for independent copies.
