# Develop through the agent interface

The supported loop is **edit → validate → save → reload → control → inspect → capture**. Tools use the game's authoring functions and runtime. Inspection is read-only; input and stepping are temporary controls. Saving changes an actual source file and returns its revision.

## Start a session

For Rust development, the recommended entry point is `cargo xtask agent watch`. It automatically builds, serves, and replaces browser sessions as source changes; [Rust iteration](rust-iteration.md) explains build diagnostics, session transitions, and recovery. Use the direct Node client while Rust is broken, since Cargo may be unable to compile its CLI dependencies.

For an independently managed server and browser, run `. ./scripts/activate.ps1` in each PowerShell terminal, then:

```sh
cargo xtask dev
```

```sh
cargo xtask agent serve --url http://127.0.0.1:5173/playground.html --port 7331
```

Keep both running. The bridge prints JSON with its session ID, endpoint, and game URL. It uses installed Chrome by default; use `--browser PATH` or `BROWSER_CHANNEL` for another Chromium browser. The session starts paused with the scene drawn. Ordinary game URLs continue to run normally.

Discover sessions from another terminal:

```sh
cargo xtask agent sessions --port 7331
```

Every command identifies its session. To avoid shell quoting differences, write a command file:

```json
{"op":"inspect","entity":"player","component":"Transform"}
```

```sh
cargo xtask agent request --port 7331 --session SESSION_ID --file command.json
```

Responses include request/session IDs, frame, tick, and scene ID/revision. Counters and revisions are decimal strings to preserve all 64-bit values. Errors have stable `code`, `path`, and `message` fields and a nonzero CLI exit status. Do not automatically retry a mutation after a lost response.

## Discover and observe

Submit these objects through `agent request`, or use `agent <op> --json ...`:

| Command | Behavior |
| --- | --- |
| `{"op":"describe"}` | List registered projections and schemas |
| `{"op":"describe","name":"Camera"}` | Generated view schema, defaults, units, constraints |
| `{"op":"describe","name":"BrowserControl"}` | Generated control request schema |
| `{"op":"describe","name":"SceneDocument"}` | Authoring schema and validation constraints |
| `{"op":"status"}` | Frame/tick/revision, state, budgets, render observations |
| `{"op":"inspect","limit":20,"offset":0}` | Page through live authored entities and names |
| `{"op":"inspect","entity":"player","component":"Transform"}` | Read selected component data |
| `{"op":"inspect","entity":"player-accessory","component":"GlobalTransform"}` | Read a fresh world matrix without advancing gameplay |
| `{"op":"inspect","entity":"player","component":"PresentedTransform"}` | Read the matrix prepared at the last extraction boundary |
| `{"op":"inspect","component":"PresentationState"}` | Read extraction mode, fraction, source tick and history storage |
| `{"op":"inspect","parent":"west-lantern","component":"Hierarchy","limit":20}` | Page direct children and their prefab identities |
| `{"op":"inspect","component":"Camera"}` | Read the shared camera resource |
| `{"op":"inspect","component":"Environment"}` | Read the whole look as a scene `environment` |
| `{"op":"inspect","component":"Shadows"}` | Read one look section; also `Atmosphere`, `AmbientOcclusion`, `Bloom`, `LocalLights`, `SunShafts`, `Reflections`, `Particles` |
| `{"op":"inspect","entity":"lamp","component":"LocalLight"}` | Read an entity's light or `ParticleEmitter` as its scene version 6 field |
| `{"op":"inspect","component":"GameProgress"}` | Read the sample's score/jump resource |
| `{"op":"assets"}` | CPU residency; GPU residency accompanies results |
| `{"op":"culling","enabled":false}` | Temporarily toggle frustum culling and render while paused |
| `{"op":"camera","position":[0,31,64],"target":[0,0,0],"vertical_fov_radians":0.96,"near":0.1,"far":1000}` | Set and render a validated temporary camera |
| `{"op":"environment","atmosphere":{"preset":"night"},"bloom":{"intensity":0},"sun_shafts":{}}` | Replace the named look sections and render while paused |
| `{"op":"benchmark","warmup":30,"samples":120,"mode":"step"}` | Sample paced frames, CPU timings, counters and asset memory |
| `{"op":"describe","name":"ProfileCapture"}` | Discover optional detailed profiling availability and bounds |
| `{"op":"profile","duration_ms":5000,"max_events":65536,"cpu_samples":true}` | From a paused profiling session, capture ordinary frames, save trace/CPU-profile artifacts, then pause |
| `{"op":"input","forward":1,"held":1,"pressed":1}` | Replace temporary input; bit zero is jump |
| `{"op":"step","ticks":30}` | Execute exactly 30 fixed ticks and render |
| `{"op":"present","alpha":0.5}` | Preview the midpoint of retained fixed-tick poses without advancing gameplay |
| `{"op":"capture"}` | Save a browser canvas PNG with its observation |
| `{"op":"capture","alpha":0.5}` | Render and capture an explicit fractional pose |
| `{"op":"capture","alpha":0.5,"gpu":true}` | Capture the same fraction through GPU presentation (decision 014) |
| `{"op":"resume"}` / `{"op":"pause"}` | Change execution state |
| `{"op":"diagnostics"}` | Read bounded browser warnings/errors |
| `{"op":"close"}` | Close the owned browser and bridge |

Input, step, render, capture, reload, and the camera, culling and environment controls require a paused session. Input defaults to zero; each command replaces the previous batch. Axes/held bits persist across steps. Press/release edges and look deltas are consumed by the next step only. Resume, suspension, focus loss, and successful reload clear controls. Repeating pause while paused preserves them.

Step accepts 1–120 ticks, bypasses catch-up limits, runs one presentation phase, and increments the frame once. Presentation time is capped at 250 ms. Use repeated one-tick requests for behavior depending on presentation every tick. Exact tick count does not guarantee cross-platform floating-point determinism.

Capture writes under `artifacts/agent/SESSION_ID/` without advancing simulation. It reports the rendered frame/revision. Default step/render/capture show exact current gameplay state, even after a fractional preview. Explicit `present` and optional capture `alpha` require finite values in [0, 1] and a paused session. Interpolation applies only to selected local transforms and their inherited hierarchy motion. Observation `presentation` records the last prepared extraction; its frame/tick can differ from the authoritative observation. Read `Transform`/`GlobalTransform` for gameplay and `PresentedTransform` for the prepared pose. Ordinary running frames present on the GPU ([decision 014](adr-014-gpu-presentation-and-culling.md)); `presentation.gpu_pending` then marks CPU poses as not yet prepared, and inspecting `PresentedTransform` prepares that frame's exact CPU reference first. Add `"gpu":true` to `present` or an alpha `capture` to preview the GPU path; its visible count comes from an earlier GPU frame's read-back. CPU render timing measures submission, not GPU execution. If rendering fails after stepping, simulation may already have advanced; the error includes the resulting observation.

## Save and reload

Read the source path returned by the game. Prepare a candidate retaining its scene ID and incrementing revision exactly once:

```sh
cargo xtask scene validate --file candidate.scene.json
cargo xtask scene save --file web/public/scenes/playground.scene.json --candidate candidate.scene.json --expected-revision 1
```

Save validates, checks the revision, serializes cooperating writers with a sidecar lock, and atomically replaces the file with candidate bytes. Rejection preserves the original. Success reports `persisted:true`. Direct editors do not participate in the lock; avoid simultaneous direct writes. [Authoring ownership](../crates/engine_dev/README.md) covers crash recovery.

Reload the saved revision:

```json
{"op":"reload","expected_revision":"1","revision":"2"}
```

The browser checks its active revision and the fetched scene ID/revision before replacement. Validation/admission failure preserves the game. Successful reload resets gameplay and renders the accepted revision. It reports `saved_by_reload:false` because it reads source. Source saves do not trigger Vite page reloads. Rebuilt/replaced runtimes require a fresh agent session.

For isolated workflows, development URLs accept `?scene=/scenes/<name>.scene.json`. Production disables this selector. The browser smoke uses its own fixture and leaves the sample source untouched.

## Change the look

The look is a set of engine resources: `Atmosphere` (sun, sky, fog, grade),
`Shadows`, `AmbientOcclusion` and `Bloom`
([decision 022](adr-022-scene-environments-and-agent-controls.md)), and
`LocalLights`, `SunShafts`, `Reflections` and `Particles` (decisions
[023](adr-023-local-lights.md) to [026](adr-026-transparency-and-particles.md)).
Inspect `Environment` for all of them in the shape of a scene
`environment`; `describe` gives each section's schema and defaults. An
atmosphere is a preset (`day`, `golden_hour`, `dusk`, `night`) plus the fields
that differ from it. Its grade sets the palette
([decision 027](adr-027-stylized-grading.md)): `exposure`, `contrast`,
`saturation`, `vibrance` (color for muted tones) and the split-toning gains
`shadow_tint` and `highlight_tint`; fog (`fog_start`, `fog_density`,
`fog_max`) is what most often grays a large scene. Sun shafts, reflections and refraction are opt-in: missing, they
read as intensity (or strength) 0, and an empty section (`"sun_shafts":{}`) turns them on
with their defaults.

Try a look with the `environment` control. It takes the same sections,
validates them like a scene (`invalid_environment` with the field's path,
changing nothing), replaces only the sections it names, renders, and lists
them in `environment_changed`. It never changes source. To keep a look, put
the sections in the scene's `environment` (version 5; version 6 for the
local light, shaft, reflection and particle sections and the atmosphere's
grade fields; version 8 for refraction), save and reload: the
scene then sets every section, omitted ones taking the engine's default state
(sun shafts, reflections and refraction off), and replaces any temporary look. Scenes
without an environment leave the look to the game's code. Compare looks in
captures at identical canvas sizes.

Lights, particles and glass are scene data too. Version 6 entities author a
`light` (color, intensity, range, local `offset`, optional `spot` with
angles in degrees) and `particles` (rate, lifetime, speed, `spread_degrees`,
acceleration, size and color at birth and death, `blend` `alpha` or
`additive`, ...); materials author `opacity`, and version 8 materials
`transmission`, `ior` and `thickness` for glass that the environment's
`refraction` bends and blurs. A light on an entity with a mesh
follows its displayed pose. Particles are a function of presentation time,
so paused captures are exact and repeatable.

Every reply's `rendering` reports the effects the last frame ran and its
shadow work: `shadows_enabled`, `shadow_draws` (beside the camera pass's `draw_calls`),
`shadow_kept_redraw` (0 none, 1 one strip, 2 every still caster),
`shadow_near_casters`, `shadow_moving_casters` and `shadow_kept_casters`
(read back a few frames late, null until then), `ambient_occlusion` and
`bloom`; `local_lights` and `dropped_lights`, with `lights_visible` and
`light_cluster_overflow` read back late (overflow means shorter light ranges
are needed); `sun_shafts`, `reflections` and `refraction`; `transparent_draws` and
`particles`. `Mesh3d` reports each material's `surface`, `opacity` and
whether its texture masks emission (`texture_emission_mask`).

## Extend a game

Enable `engine/dev`. Register typed component/resource projections with `engine::dev::DevTools` and `InspectorDescriptor`. Each supplies bounded JSON and metadata; arbitrary ECS memory is never exposed. Pass the active `SceneInstance` for authored entity queries. Resource inspection also works in procedural games.

`engine::dev_browser::DevBrowser` provides input, step, render, and inspection handling for Wasm clients. Both examples show its small adapter. Crystal Garden adds `GameProgress` and revision-checked reload. New projections require no central registry edits. Callbacks are trusted game code and must bound their work/allocations.

Production (`cargo xtask build`) generates `web/pkg` with development disabled. Development generates separate `web/pkg-dev` bindings. The host imports its adapter only in Vite development mode with `engine_dev=1`. Built-in view schemas come from serialized Rust DTOs at build time; reflection/schema generation stay out of the Wasm frame loop.

Static textured GLB import is supported through `cargo xtask asset import`; [asset import](asset-import.md) documents subset limits, namespace replacement, and revision-checked persistence. `assets` includes authored texture IDs, CPU readiness and browser GPU residency, while render observations report texture upload/residency bytes. `Mesh3d` inspection reports texture readiness and dimensions. Pixels and runtime handles never appear in these observations.

Current limits: registered projections and authored entities only; no general component mutation, runtime-spawned entity browsing, project scaffolding, live shader replacement, or MCP adapter. Watched Rust/WGSL changes rebuild Wasm and create a fresh runtime; they do not preserve temporary gameplay state. Coding pipelines can use CLI/HTTP directly.

Use [the benchmark runner](benchmarking.md) for reproducible A/B comparisons and
stored JSON baselines. A single session's `benchmark` is a temporary control: it
clears input and advances one fixed tick per warm-up/sample in `step` mode, or
no gameplay in `render` mode. It requires a paused, visible canvas and aborts on
resize/disposal or a ten-second deadline. Limits are 0–120 warm-up frames and
1–240 samples. Incomplete runs return structured failure, not partial percentiles.
Do not retry uncertain requests; their completed steps may have advanced gameplay.

For deeper CPU/GPU/collision/memory/browser investigation, build `cargo xtask
profile` and serve `web/dist-profile` separately. Agent `profile` requires this
profiling Stress Lab, clears input, resumes ordinary frames for 100-5,000 ms,
then pauses. It returns files under `artifacts/agent/SESSION_ID/`; `cpu_samples`
defaults to true and adds a separate CDP `.cpuprofile`. This instrumented capture
reports completeness and capability gaps. Follow [profiling](profiling.md) for
setup, limits and comparison rules.

Rendering replies report `render_candidates`, `visible_instances` before the
instance budget, `culled_instances`, submitted `instances`, `dropped_instances`,
`missing_assets` and `invalid_transforms`. `Mesh3d.local_bounds` explains the
immutable geometry bounds used for culling. Off-screen resources can remain GPU
resident while their valid renderables still reference them; visibility is not
asset disposal. Production excludes benchmark sampling and temporary controls.

Scene version 3 supports prefabs and parent transforms. Edit a placement's local
`transform` to move its entire subtree; edit its definition to change every
placement on the next validated reload. Expanded IDs such as
`west-lantern.lantern` are inspectable, while source edits use the definition's
local IDs. `Hierarchy` includes both source and current parent identities.
`Transform` is local; `GlobalTransform` returns a column-major world matrix,
preserving shear from rotated children under nonuniform parent scales. Invalid
runtime branches report `valid:false` and are counted in
`rendering.invalid_transforms`. The [hierarchy contract](adr-006-hierarchy-and-prefabs.md)
defines bounds, sharing and disposal. Import reusable definitions with
`asset import --as-prefab`, then place them explicitly in scene JSON.

Rust gameplay should use `Query<&mut Transform>` for frequent local motion.
`Transform` remains an ordinary mutable Bevy component. Read global poses through
`Res<GlobalTransforms>` and presented poses through `Res<PresentedTransforms>`;
the derived read values are no longer components. `Parent` stays immutable; use checked
parent helpers. `edit_transform` performs an immediate edit without insertion
observers. `TransformEdits` still queues replacements with ordinary lifecycle
observers; chain systems that must read those deferred edits.

Propagation scans change metadata against an engine-owned watermark independent
of `clear_trackers`, then advances the ECS change tick after refresh. Repeated
direct edits are visible without advancing frame/fixed clocks. In-place motion
does not emit `Insert`/`Discard`; use the named propagation boundaries or retained
store output revisions. Explicit `bypass_change_detection` and manual tick edits
opt out and are unsupported for transform updates. Interpolation captures changed
local sources during the same validation pass. See the
[mutation contract](adr-011-in-place-transforms.md) and
[canonical pose migration](migrations/0.1-canonical-poses.md) before updating a gameplay system.

## Verify

```sh
cargo xtask lint
cargo xtask test
cargo xtask build
cargo xtask docs
node web/scripts/agent-smoke.mjs
```

The last command requires the development server and Chrome. It verifies inspection, input/step/capture, the environment control, guarded saves, and reloads, including a saved scene environment. [Bridge details](../web/scripts/agent.README.md) specify bounds, deadlines, and recovery.

## Collision observations

Discover `Collider`, `CollisionWorld` and `DevelopmentProtocol`. Spatial commands
refresh authoritative transforms and the collision cache without advancing ticks,
consuming input, changing the presented pose, or consuming sensor events.

```json
{"op":"raycast","origin":[0,10,0],"direction":[0,-1,0],"max_distance":20,"filter":{"memberships":2,"filter":1,"sensors":"exclude"}}
```

In Crystal Garden, world/player/pickup membership bits are 1/2/4. Filtering is
symmetric, so the query membership must also match the target's filter.
`raycast` and `shape_cast` return the closest hit or null. `overlap` accepts a
world position, optional quaternion rotation, tagged cuboid/sphere shape, filter,
and result limit (1–100). Shape casts add direction and maximum distance; their
orientation stays fixed. Geometry targets can have full affine parent transforms.

`collision_events` accepts optional authored entity, offset and limit. It returns
a non-consuming page from the latest attempted fixed tick, plus completeness,
committed tick and sequence. A multi-tick step retains only its final tick's
events; step one tick when observing individual transitions. Incomplete sensor
attempts retain previous committed pairs and emit no partial transition list.

Hits expose authored identity when available and explicitly mark runtime-only or
removed entities. A missing plugin, invalid geometry, incomplete cache or query
budget exhaustion returns a structured error. An error is not evidence of an
empty volume or clear movement path. Inspect `CollisionWorld` for cache bounds
and diagnostics. Colliders are saved through scene version 4 using the existing
revision-checked workflow; runtime queries never persist changes.

Run `node web/scripts/collision-smoke.mjs` against fresh development Wasm to
verify the same interfaces in Chrome. See [collision contracts](adr-009-collision-and-spatial-queries.md).
