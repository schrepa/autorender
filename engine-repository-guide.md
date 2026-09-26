# Repository guide for a Rust browser game engine

Rust · WebAssembly · wgpu and WebGPU · WGSL · thin TypeScript host

Research reviewed 24 September 2026. These are recommended repository policies, informed by the primary sources below; they are not a claim that one architecture is universally fastest. Testing practices are intentionally outside scope.

## 1 Choose the foundation deliberately

**Recommended starting point:** own the game API, rendering architecture, asset pipeline, and browser integration; reuse `wgpu` and established math, serialization, and decoding libraries. Start by evaluating `bevy_ecs` and `bevy_app` as components. Write an ECS or scheduler from scratch only when a documented requirement conflicts with them.

If game systems expose Bevy `Query`, `Res`, or component traits, Bevy is part of your public compatibility contract. Re-exporting them does not hide that coupling. Choose either an explicitly Bevy-based SDK with coordinated upgrades, or an independent SDK whose adapters keep those types private. Record this choice in the first architecture decision record. Avoid a permanent whole-engine fork unless the team accepts maintaining upstream divergence. [3, 5]

## 2 Organize around ownership

Use one Cargo workspace and a small set of crates. Create a crate for an enforceable dependency boundary, platform boundary, or reusable subsystem; use modules for ordinary organization.

| Location | Responsibility |
| --- | --- |
| `crates/engine/` | Public façade, small prelude, default configuration, subsystem composition |
| `crates/engine_core/` | Scheduling contracts, gameplay data, typed IDs and service interfaces |
| `crates/engine_render/` | Render extraction, WGSL, pipelines, GPU resource ownership |
| `crates/engine_platform/` | Input, clocks, storage, asset transport, task execution and lifecycle adapters |
| `web/` | TypeScript bootstrap, DOM integration, worker startup and deployment configuration |
| `tools/assetc/` and `tools/xtask/` | Offline asset processing and reproducible repository commands |
| `examples/` and `docs/` | SDK examples, architecture decisions, subsystem contracts and migrations |

**Dependency rule:** `engine` composes the subsystems; render and platform depend on narrow core contracts; core never imports either implementation. Cross-subsystem wiring belongs in composition code. Keep browser-specific rendering glue localized. Add a separate asset crate when its ownership warrants it; do not let core become a miscellaneous utilities crate. Editors and asset tools are engine clients, never runtime dependencies.

Each subsystem gets a short README stating its owner, public entry points, data lifetime, execution context, dependencies, and invariants. Assign corresponding paths in `CODEOWNERS`.

## 3 Make extension easy and costs visible

**Provide three intentional surfaces:** ordinary gameplay APIs; supported extension points for plugins, materials, loaders, and render passes; private implementation. Keep raw `wgpu` objects and JavaScript values out of ordinary gameplay. Low-level render extensions may expose them deliberately, with an explicit version policy.

Prefer typed handles, builders with sensible defaults, and ordinary Rust traits over string registries or a new macro language. Keep fields private where future changes need freedom. Document errors, deferred effects, allocations, copies, resource ownership, and capability requirements. For example, loading a mesh returns a handle immediately; readiness and failure are separately observable. [4]

Plugins declare dependencies, configuration, resources, and named scheduling sets. Publish insertion points so third-party code can extend a phase without editing engine internals. Specify duplicate registration and removal behavior. Distinguish startup readiness from scene disposal; Bevy's `Plugin::cleanup` is a startup hook, not a runtime teardown hook. [5, 7]

Maintain a minimal game and an external-style extension example using only the public façade and documented extension APIs. A small feature should require a plugin or system, not a new central manager branch.

## 4 Put scheduling and ownership ahead of cleverness

**Write down the frame contract:** input ingestion, fixed simulation, deferred command application, presentation update, render extraction, and submission. State exactly when spawned entities and queued events become visible. Bound fixed-step catch-up after suspension; interpolate presentation where appropriate.

Express actual ordering requirements as dependency edges between named system sets. Declare narrow read/write access so independent work can overlap. Broad mutable-world access and implicit plugin insertion order undermine this design. Parallel scheduling does not itself guarantee deterministic simulation. [6]

Use typed generational IDs for runtime references; keep borrowed component data within its valid access scope. Give each mutable dataset one owner instead of sharing the world behind a global mutex. Prefer contiguous batches and access-pattern-specific layouts; structure-of-arrays is a choice, not a universal rule.

For hot loops, reuse capacity, batch mutations, and avoid per-entity string lookup, heap allocation, or dynamic dispatch unless justified. Bound queues and caches, with explicit overflow or eviction behavior. Keep unsafe code in narrowly owned modules behind safe APIs, documenting aliasing, alignment, initialization, and lifetime invariants. [18]

## 5 Treat the browser boundary as a subsystem

**Keep simulation in Rust and the TypeScript host thin.** Exchange coarse commands, input batches, and asset payloads. Generate binding declarations; version worker messages and persisted formats. Avoid per-component JS calls or JSON serialization inside the frame loop.

Zero-copy views into Wasm memory require an explicit borrowing protocol. Growth of linear memory can invalidate JavaScript views, and those views do not enforce Rust lifetimes. Default to synchronous, bounded access; copy or transfer owned payloads when work outlives that access. Never retain a borrowed view across arbitrary engine calls or `await`. [12]

Keep browser GPU resources in one owning execution context, on the main thread or a supported render worker. Send CPU jobs and results between workers; do not treat GPU handles as transferable. `wgpu` explicitly warns that browser GPU objects cannot be shared across threads. [10]

Provide one gameplay contract with a single-thread executor and an optional worker executor. Async work is not automatically parallel. Shared-memory Wasm threading needs cross-origin isolation and a dedicated build configuration; the current `wasm-bindgen-rayon` route also requires a pinned nightly toolchain and rebuilt standard library. Bevy's Wasm threading tracker remains open. Do not assume native parallelism carries over. [13, 14]

Keep filesystem and thread creation behind platform adapters. Declare SIMD requirements per Wasm artifact; select a compatible artifact before instantiation when supporting a fallback. Unsupported instructions cannot safely be hidden behind an ordinary runtime branch. [19]

## 6 Give rendering its own data and lifecycle

**Separate gameplay state from render state.** Extract only necessary data into an owned snapshot or explicitly synchronized buffers. Let the renderer classify and batch draws, prepare uploads, and encode commands. This isolates GPU changes from gameplay and permits pipelining later; extraction need not clone the world or imply multithreading. Bevy's distinct rendering stages are useful inspiration, not a fixed phase order to copy. [9]

Keep CPU components distinct from GPU buffer records. Co-locate WGSL, binding layouts, and upload schemas. Use a WGSL-aware layout library such as `encase`; `repr(C)` or plain byte-cast eligibility alone does not establish the required shader layout. [11]

Centralize pipelines, bind groups, upload buffers, and resource retirement. Cache keys include shader version, layout, render formats, and specialization options. Batch uploads; begin with queue writes before introducing custom staging machinery. Keep GPU-generated data on the GPU where practical, and avoid frame-loop waits for readback. [16]

Expose bounded render-extension hooks with declared inputs, outputs, phase, and capability requirements. Introduce a render graph when pass dependencies and resource lifetimes justify it. Centralize device loss recovery and resource reconstruction; preserve logical asset handles while replacing GPU backing resources. Retain reload recipes or source locations, with an explicit CPU-memory budget. [17]

## 7 Make assets and mobile capability explicit

Separate authored sources, versioned cooked assets, and disposable caches. An asset cache key includes source content, dependencies, importer version, settings, and target tier. Keep expensive mesh and texture processing in `assetc`; stream runtime-ready data in bounded chunks.

Use stable authored identifiers for scenes and saves; runtime entity indices and asset handles are not persistent IDs. Specify asset states such as loading, CPU-ready, GPU-ready, failed, and evicted. Document strong versus weak ownership, cancellation, dependency failure, and when reloads become visible. Bevy's asset system provides a useful model for asynchronous loading and typed handles. [8]

**Separate build features from runtime capabilities.** Define a minimum supported WebGPU profile and higher quality tiers. Query adapter capabilities and request only the features and limits needed; unavailable requirements can make device creation fail. Choose a clear unsupported-device response. WebGL fallback is a separate renderer scope, not a free switch. [10, 15]

Mobile tiers control texture residency, mesh detail, shadows, post-processing, draw density, and resolution. Put budgets in configuration, not scattered platform checks. Centralize resize, suspend/resume, input adaptation, and audio activation. Share gameplay APIs across tiers; equivalent desktop fidelity is not the mobile contract.

## 8 Make the repository enforce the design

**Reproducible builds.** Commit `Cargo.lock`, the JavaScript lockfile, and `rust-toolchain.toml`. Pin build tools and the matching `wasm-bindgen` CLI. Record the supported Rust version and browser capabilities. Keep a stable baseline; isolate any nightly threading path. Centralize dependency declarations, profiles, and lints at the workspace root; member crates must explicitly opt into inherited dependencies and lints. For a new Rust 2024 virtual workspace, set resolver 3 explicitly. [1]

**Controlled features.** Keep default features small and additive: enabling another feature must not break existing behavior. Use target-specific dependencies for platform selection; avoid mutually exclusive `web` and `native` feature pairs. Inspect duplicate foundational dependencies and feature expansion with `cargo tree -d` and `cargo tree -e features`. Public dependency upgrades receive migration notes. [2, 3]

**One command vocabulary.** Implement `cargo xtask dev`, `build`, `lint`, `docs`, and `assets` as the repository's documented entry points. Local use and automation invoke the same commands. Publish named package/target/feature presets for browser, optional threaded browser, and native development; do not try to build host-only tools for Wasm with one indiscriminate workspace command. Automate formatting, Clippy, supported-preset compilation, and rustdoc generation. These command names are a proposed interface, not existing tools.

**Small change contracts.** Every subsystem-changing PR identifies the owner, public API impact, memory/copy/synchronization costs, capability assumptions, and migration needs. Change examples and documentation alongside APIs. Use short architecture records for dependency direction, threading, asset formats, or renderer extension changes. Document deprecations before removals; keep format migrations separate from runtime IDs.

**Operational visibility.** Use consistent tracing spans and GPU labels, with counters for queue depth, upload bytes, resident resources, and cache growth. Keep these inexpensive or optional. Ignore generated packages, cooked caches, and build output in Git; version source assets deliberately, using large-file storage when needed. Preserve licenses and attribution when adopting upstream code.

## 9 Design the developer workflow for AI coding agents

Treat agent usability as a supported developer interface. Retain the Rust/Wasm/WebGPU foundation and ownership rules above; make discovering APIs, editing a game, and observing its behavior possible through documented commands.

**Discoverable SDK.** Publish searchable, version-specific signatures, component schemas, defaults, constraints, and small examples. Generate structural metadata from code. Keep a short `AGENTS.md` as a map to subsystem instructions and repository-local documentation. Give common tasks one recommended approach; expose advanced options through explicit extension points. [20]

**Editable authoring.** Give scenes, prefabs, materials, and input maps canonical text representations with schemas, stable authored identifiers, and readable diffs. Route editor and agent operations through the same authoring functions. A tool must distinguish persistent source edits from temporary runtime changes and report the resulting revision; restarting the game must not silently discard a change presented as saved.

**One development interface.** Put development operations in a separate crate, such as `engine_dev`, with a CLI and optional MCP adapter sharing its implementation. Support noninteractive project creation, asset imports, scene launching, and inspection. In browser builds, connect a development client to a local bridge. Version the protocol and identify sessions. Proposed commands include `engine api describe Camera`, `engine inspect entity player`, and `engine capture`.

**Observe the running game.** Expose filtered component queries, asset readiness, rendering diagnostics, input control, pause/step operations, and screenshots from the actual browser build. Attach frame or revision identifiers to observations. Include readable names alongside IDs and provide links to source assets where available. Bevy's Remote Protocol offers useful reference mechanisms for ECS inspection, schema discovery, and change subscriptions. [22]

**Actionable feedback.** Return structured errors with a stable code, affected entity or asset, source or field location, violated constraint, and relevant accepted values. Keep normal results concise; support field selection, pagination, and expanded detail on request. An agent inspecting one camera should not receive the entire scene or an unbounded log stream. [21]

**Local changes and fast iteration.** Keep gameplay features in focused modules so adding an ability rarely requires changing engine internals or several global registries. Maintain complete examples for cameras, movement, UI, and asset loading. Support asset and shader reload plus incremental Rust builds. Give each working copy independent development sessions, ports, and output paths. [20]

**Preserve the performance contract.** Generate schemas during builds and enable inspection through an optional development plugin. Process mutations at documented scheduling boundaries; bound snapshot sizes and inspection work. Keep reflection, serialization, and protocol traffic off ordinary per-entity execution paths. Compile gameplay into Rust systems and efficient GPU operations. Prioritize the discoverable SDK, editable scene format, and inspection bridge before adding a large tool catalogue.

## Primary sources

Sources support the mechanisms and constraints cited above; the proposed repository layout and policies are the author's synthesis. Library documentation links may advance, so record the versions adopted by the actual repository.

1. [Cargo workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html) — inheritance, resolver, shared configuration.
2. [Cargo features](https://doc.rust-lang.org/cargo/reference/features.html) — feature unification and additive design.
3. [Cargo SemVer compatibility](https://doc.rust-lang.org/cargo/reference/semver.html) — public API and dependency compatibility.
4. [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/checklist.html) — predictable, documented interfaces.
5. [Bevy plugin development](https://bevy.org/learn/quick-start/plugin-development/) — modular dependencies and extension conventions.
6. [Bevy ECS scheduling](https://docs.rs/bevy_ecs/latest/bevy_ecs/schedule/index.html) — ordering and access conflicts.
7. [Bevy Plugin lifecycle](https://docs.rs/bevy_app/latest/bevy_app/trait.Plugin.html) — startup phases and cleanup semantics.
8. [Bevy assets](https://docs.rs/bevy_asset/latest/bevy_asset/) — asynchronous loading and handle ownership.
9. [Bevy rendering stages](https://docs.rs/bevy_render/latest/bevy_render/enum.RenderSystems.html) — render subsystem phases.
10. [wgpu crate documentation](https://docs.rs/wgpu/latest/wgpu/) — backends, features and browser thread restrictions.
11. [encase](https://docs.rs/encase/latest/encase/) — Rust data and WGSL buffer layout.
12. [js-sys typed array views](https://docs.rs/js-sys/latest/js_sys/struct.Uint8Array.html#method.view) — memory and lifetime constraints.
13. [wasm-bindgen-rayon](https://github.com/RReverser/wasm-bindgen-rayon) — workers, isolation and build requirements.
14. [Bevy Wasm threading tracker](https://github.com/bevyengine/bevy/issues/4078) — unresolved integration work.
15. [wgpu DeviceDescriptor](https://docs.rs/wgpu/latest/wgpu/type.DeviceDescriptor.html) — requested capabilities and limits.
16. [WebGPU buffer uploads](https://toji.dev/webgpu-best-practices/buffer-uploads) — implementation guidance from WebGPU contributor Brandon Jones.
17. [WebGPU device loss](https://toji.dev/webgpu-best-practices/device-loss) — device and resource recovery, Brandon Jones.
18. [Rustonomicon safe and unsafe interaction](https://doc.rust-lang.org/nomicon/safe-unsafe-meaning.html) — safety boundaries.
19. [Rust browser Wasm target](https://doc.rust-lang.org/rustc/platform-support/wasm32-unknown-unknown.html) — host limitations and instruction compatibility.
20. [OpenAI harness engineering](https://openai.com/index/harness-engineering/) — repository knowledge, focused instructions, and observable development environments.
21. [Anthropic tool design for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) — clear contracts, concise results, and actionable errors.
22. [Bevy Remote Protocol](https://docs.rs/bevy_remote/latest/bevy_remote/) — ECS inspection, type schemas, and remote method discovery.
