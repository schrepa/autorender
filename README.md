# Browser Engine

A browser-first Rust game engine with Bevy-based gameplay, a WebGPU renderer, typed mesh/material/texture assets, and versioned authored scenes. The engine owns its frame loop, asset lifetimes, and browser integration.

Two independent clients exercise the public SDK: a 1,000 / 10,000 / 50,000 cube workload and **Crystal Garden**, an authored game with movement, jumping, collectibles, custom geometry, and scene reload. Both use the same runtime, renderer, input host, and diagnostics.

**Stress Lab** at [`/stress.html`](http://127.0.0.1:5174/stress.html) opens with 90,000 textured towers; the Wild city preset builds a procedural city of 52 shapes with traffic, trains, drones, cranes and Ferris wheels. Change rendering, motion, hierarchy, interpolation, collision and query loads independently; record ten seconds and export timings with configuration, seed, camera and artifact identity. See [Stress Lab](docs/stress-lab.md). It uses the same public SDK, with no stress-specific engine code.

Use **`cargo xtask profile`** for a separate Stress Lab build with detailed CPU, GPU, collision, memory and browser capture. [Profiling](docs/profiling.md) documents the UI, agent capture, feature isolation and measurement limits.

**Agent-driven development is the primary workflow.** The optional interface supports identified browser sessions, schema discovery, bounded inspection, exact stepping/input, screenshots, and revision-checked scene saves/reloads. Start with the [agent workflow](docs/agent-workflow.md).

For automatic Rust iteration, run **`cargo xtask agent watch`**. It owns a web server, watches source changes, builds Wasm, and activates a fresh paused browser session after each successful build. Failures report structured diagnostics and preserve the running game. See [Rust iteration](docs/rust-iteration.md); query its status with `node web/scripts/agent.mjs development`.

For static textured models, use **`cargo xtask asset import`**. The native importer converts static GLB models with glTF metallic-roughness materials into scene version 8 and the image and mesh files beside it, with structured errors, explicit namespace replacement, and revision-checked saves. Add `--as-prefab` to create a reusable definition. See [asset import](docs/asset-import.md).

## Run

Prerequisites: Rust via [rustup](https://rustup.rs), Node.js 24 or later, and a WebGPU-capable browser. Windows Rust builds also require Visual Studio C++ Build Tools. `rust-toolchain.toml` pins Rust 1.98.1 and the Wasm target.

```sh
cargo install wasm-bindgen-cli --version 0.2.127 --locked
cargo xtask dev
```

Open **http://127.0.0.1:5173**. Drag the scene to look, use **WASD** or arrow keys to move, and **Q/E** to descend/ascend. The canvas must have keyboard focus; click it first. Touch dragging changes the view. Use Reset view to return to the starting camera.

Open **http://127.0.0.1:5173/playground.html** for Crystal Garden. Click the scene, move with **WASD/arrows**, and press **Space** to jump. Collect all six crystals. Touch devices show movement buttons.

In the prepared Windows workspace, Rust and wasm-bindgen were installed under ignored `.tools` without changing your global PATH. Activate them in each PowerShell session:

```powershell
. ./scripts/activate.ps1
cargo xtask dev
```

`dev` builds release Wasm with development support into `web/pkg-dev`. Production `build` uses separate `web/pkg` bindings without that feature. Vite updates host files; rerun the command after Rust changes. WebGPU requires a secure context, which localhost provides.
Use `cargo xtask dev --port 5174` for another working copy. The server fails clearly if its chosen port is already occupied.

## Commands

| Command | Result |
| --- | --- |
| `cargo xtask dev` | Build Wasm and serve the demo |
| `cargo xtask agent watch` | Watch Rust, rebuild, and replace managed browser sessions automatically |
| `node web/scripts/agent.mjs development` | Read build status and compiler diagnostics, even while Rust fails to compile |
| `cargo xtask agent serve --url http://127.0.0.1:5173/playground.html` | Launch an identified, paused browser session |
| `cargo xtask agent sessions` | Discover sessions on the selected bridge port |
| `cargo xtask scene validate --file PATH` | Validate source with structured JSON results |
| `cargo xtask asset import --file MODEL.glb --id NAME --output NEW.scene.json` | Cook a static textured model into a new validated scene |
| `cargo xtask build` | Production bundle in `web/dist` |
| `cargo xtask profile` | Optional profiling Stress Lab in `web/dist-profile`; ordinary game bundle remains separate. `--attribution` builds `web/dist-attribution` with hot helpers out of line for function sampling |
| `cargo xtask lint` | Formatting, native/Wasm Clippy, TypeScript |
| `cargo xtask test` | Rust contracts, schema/shader validation, game and host input tests |
| `cargo xtask assets` | Validate saved scene sources with structured errors |
| `cargo xtask benchmark` | Run fixed browser workloads with culling on/off and emit comparable JSON |
| `cargo xtask schema` | Generate the versioned scene JSON Schema from Rust types |
| `cargo xtask docs` | Native and Wasm Rust API reference in `target/doc` and `target/wasm32-unknown-unknown/doc` |
| `npm run smoke --prefix web` | Browser integration checks against a running server |
| `node web/scripts/agent-smoke.mjs` | Verify the agent loop with an isolated source fixture |
| `node web/scripts/iteration-smoke.mjs` | Verify Rust failure/recovery, changed gameplay, session replacement, and cleanup |
| `node web/scripts/texture-smoke.mjs` | Verify texture readiness, visible replacement, upload reuse, failure preservation, and retirement |
| `node web/scripts/hierarchy-smoke.mjs` | Verify parent motion, prefab placement, agent discovery and shared asset residency in Chrome |
| `node web/scripts/culling-smoke.mjs` | Verify camera-edge visibility, equivalent rendered pixels and GPU residency across camera changes |
| `node web/scripts/collision-smoke.mjs` | Verify queries, layers, transformed colliders, sensor events, and game movement in Chrome |
| `node web/scripts/interpolation-smoke.mjs` | Verify intermediate parent/child poses, exact agent captures, resets and presented-pose culling |
| `node web/scripts/gpu-presentation-smoke.mjs` | Compare GPU-presented Stress Lab captures and visible counts with the exact CPU path |
| `node web/scripts/taa-smoke.mjs` | Verify temporal antialiasing smooths and stays stable while captures and CPU presentation stay exact |

Commit both lockfiles when creating the Git repository. This directory did not initially contain a Git repository; no remote or publishing destination is configured. Build output and local tools are ignored.

## Build a game

Use `engine::prelude::*`, add components, and register systems through ordinary Bevy plugins:

```rust
use engine::prelude::*;

#[derive(Component)]
struct Spin;

fn spin(
    time: Res<FixedTime>,
    mut cubes: Query<&mut Transform, With<Spin>>,
) {
    for mut transform in &mut cubes {
        transform.rotation *= Quat::from_rotation_y(time.delta_seconds);
    }
}

struct GamePlugin;
impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(FixedSimulation, spin);
        app.world_mut().spawn((
            Transform::default(),
            Cube { color: [0.2, 0.8, 0.7, 1.0] },
            Spin,
        ));
    }
}
```

Use ordinary mutable `Transform` queries for frequent updates. Read derived poses
through `Res<GlobalTransforms>` or `Res<PresentedTransforms>` and their `get(entity)`
methods; returned `GlobalTransform`/`PresentedTransform` values are not components.
`Parent` stays immutable and uses checked `set_parent`/`set_parents` helpers.
`edit_transform` immediately edits a local pose without insertion observers;
`TransformEdits` retains deferred replacements and their ordinary observers.
Chain dependent systems before reading queued edits. Read world poses after named
propagation boundaries; retained consumers can compare store output revisions.
See the [canonical pose migration](docs/migrations/0.1-canonical-poses.md).

The engine scans component change metadata against its own watermark, independently
of `clear_trackers`, then advances the ECS tick after refresh so successive direct
edits remain visible. Gameplay clocks do not advance. Deliberate
`bypass_change_detection` or manual change-tick edits opt out and are unsupported
for transform updates. See [the mutation contract](docs/adr-011-in-place-transforms.md).

For angles, use `math::sin`, `math::cos` and `math::sin_cos` from the prelude, and `math::sin_cos4` for four at a time. They are branch-free, several times faster than the standard functions in browsers, and give the same bits natively and in every browser; glam's rotation constructors use them too. See [engine math](docs/math.md).

For collision, install `CollisionPlugin`, attach checked `Collider` components, and query `CollisionWorld` for rays, overlaps and translational shape casts. Sensor enter/exit snapshots run at fixed-tick boundaries. Scene version 4 authors the same data; agents discover and query it through the shared development interface. See [collision contracts](docs/adr-009-collision-and-spatial-queries.md).

The small rendering example is in [examples/orbit-demo](examples/orbit-demo/src/lib.rs). [Crystal Garden](examples/playground/README.md) demonstrates authored loading, owned scenes, and ordered gameplay systems. Bevy types are intentionally part of the SDK; upgrades require coordinated changes. Arbitrary Bevy plugins are not assumed compatible with this engine's custom schedules.

For reusable geometry, insert a `Mesh` and `Material` in `AssetLibrary` and attach their handles with `Mesh3d`. The library owns immutable CPU data; scene instances dispose their own assets explicitly. The renderer batches by mesh and retires unused GPU buffers. See [asset contracts](crates/engine_assets/README.md).

## Author a scene

Edit [playground.scene.json](web/public/scenes/playground.scene.json), increment its revision, run `cargo xtask assets`, then click **Reload scene** on the development server. A failed replacement keeps the current game alive. Changes to a production bundle require rebuilding first.

The [generated schema](docs/schema/scene-v3.schema.json) supports editor completion; Rust validation also checks references and budgets. Version 3 adds local parent transforms, transform-only nodes and reusable prefab placements sharing scene assets. Versions 1 and 2 remain readable. `SceneDocument` validates the same format for tools and games. See [hierarchy and prefab authoring](docs/adr-006-hierarchy-and-prefabs.md) and the [migration guide](docs/migrations/0.1-hierarchy.md).

## Scope

The engine now supplies bounded fixed simulation, tick-safe input actions, reusable camera controls, generational asset handles, arbitrary indexed geometry, per-mesh instancing, authored scenes, parent–child transforms, prefabs, conservative frustum culling, explicit disposal, and browser lifecycle handling. [Browser benchmarks](docs/benchmarking.md) compare culling on/off under fixed workloads and produce structured reports for agents. CPU timings and paced frame intervals are observations, not a performance guarantee or GPU execution time. See [visibility contracts](docs/adr-007-visibility-and-benchmarks.md) and [migration](docs/migrations/0.1-visibility.md).

Games can opt fixed-simulation motion into [render interpolation](docs/adr-008-render-interpolation.md) with `InterpolateTransform`. Separate presented transforms preserve authoritative gameplay state and compose through parent hierarchies. Agent step/capture remains exact; explicit fractional previews and captures expose intermediate poses. Standard smoothing adds one fixed tick of visual delay and bounded CPU history storage. Ordinary browser frames blend, compose and cull those poses on the GPU ([decision 014](docs/adr-014-gpu-presentation-and-culling.md)); the CPU path remains the exact reference and opt-out. Those frames also split each fixed tick across the frames around its due time ([decision 015](docs/adr-015-pipelined-fixed-ticks.md)), trading up to two frames of input latency for high-refresh frame rates; `set_tick_pipelining(false)` opts out. `set_temporal_antialiasing(true)` opts those frames into [temporal antialiasing](docs/adr-017-temporal-antialiasing.md). The `Atmosphere` resource sets the sun, sky, fog and grade of every frame ([decision 018](docs/adr-018-atmosphere-and-stylized-lighting.md)); presets range from day to night, graded for a vivid, stylized palette with contrast, vibrance and split toning ([decision 027](docs/adr-027-stylized-grading.md)). Materials carry a `Surface`: roughness and metalness for sun highlights and sky reflections, emission masked by texture alpha, and ripples for liquids ([decision 019](docs/adr-019-surface-materials-and-reflections.md)). The sun casts shadows, kept between frames for objects that stay still (`Shadows`, [decision 020](docs/adr-020-sun-shadows.md)), and `AmbientOcclusion` and `Bloom` darken creases and spread bright light ([decision 021](docs/adr-021-ambient-occlusion-and-bloom.md)); all three are on by default. Scenes author the whole look as an `environment` that agents inspect and try temporarily ([decision 022](docs/adr-022-scene-environments-and-agent-controls.md)). Point and spot lights follow the meshes that carry them ([decision 023](docs/adr-023-local-lights.md)); sun shafts and screen-space reflections are opt-in ([decisions 024](docs/adr-024-sun-shafts.md) and [025](docs/adr-025-screen-space-reflections.md)); transparent materials and stateless GPU particles blend order-independently ([decision 026](docs/adr-026-transparency-and-particles.md)).

Texture-mapped PBR materials (normal and roughness maps), general physics/audio, nested prefabs, binary asset streaming, device reconstruction, workers, a native windowed runtime, and WebGL fallback remain future work. Current imports target small opaque static models within the 1 MiB scene source budget. Device loss reports an error and requires reload. See [textured import architecture](docs/adr-005-textured-imports.md), [development architecture](docs/adr-003-agent-development.md), [asset/scene architecture](docs/adr-002-assets-and-scenes.md), [texture migration](docs/migrations/0.1-textures.md), and the original [design guide](engine-repository-guide.md).

Build, lint, test, asset validation, schema drift, and documentation commands are shared by local development and CI. See [verification details](docs/verification.md) for observed coverage and limits.
