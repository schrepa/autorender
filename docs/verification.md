# Engine milestone verification

Verified on 24 September 2026 in the Windows development workspace.

| Check | Result |
| --- | --- |
| `cargo xtask build` | Both release Wasm games, generated bindings, TypeScript, and production Vite bundle pass |
| `cargo xtask lint` | Authored data/schema drift, formatting, native all-target Clippy, production/development Wasm presets, and TypeScript pass |
| `cargo xtask test` | 180 Rust tests and 91 Node input/transport/adapter/source/iteration/benchmark checks pass |
| `cargo xtask assets` | Crystal Garden scene version 3 / revision 4 validates: 29 top-level records, one prefab, 33 expanded entities |
| `cargo xtask docs` | Native workspace and browser-specific API documentation builds |
| `npm run smoke --prefix web` | Chrome 153.0.8010.50 passes for both games against the production bundle |
| `node web/scripts/agent-smoke.mjs` | Real Chrome passes the complete agent loop with an isolated scene source |
| `node web/scripts/agent-lifecycle-smoke.mjs` | Browser replacement/navigation invalidate sessions; unavailable sessions close cleanly |
| `node web/scripts/iteration-smoke.mjs` | Real Rust error/repair preserves the old runtime, replaces the session after success, verifies changed gameplay in Chrome, and releases owned services |
| `node web/scripts/texture-smoke.mjs` | Chrome verifies per-ID texture readiness, visible pixel replacement, failed reload preservation, upload reuse, stable residency and retirement |
| `node web/scripts/hierarchy-smoke.mjs` | Chrome verifies world transforms, inherited player motion, stable prefab identity, shared residency, rejected reload preservation and subtree removal |
| `node web/scripts/culling-smoke.mjs` | Four on/off image comparisons have zero changed pixels; first-visible upload, offscreen retention, return without upload and removal retirement pass |
| `node web/scripts/interpolation-smoke.mjs` | Ordinary fractional frames and explicit previews agree; parent/child motion, exact captures, reset/reload identity, bounded storage and presented-pose culling pass |
| `node web/scripts/gpu-presentation-smoke.mjs` | GPU-presented and exact CPU captures of three fractions in two views differ in at most 5 of 1,080,000 pixels; visible counts are equal; counters partition candidates; inspection materializes the pending frame |
| `node web/scripts/taa-smoke.mjs` (26 September) | Captures with temporal antialiasing on, off and after temporal frames are identical; with texture mipmaps, converged frames lower total variation from 28.8 to 24.4 per pixel with 8×8 block means within 0.54 levels; frames 300 ms apart differ beyond 24 levels in 42 of 471,064 pixels; CPU-presented frames match captures exactly |
| `cargo xtask benchmark` / direct Node runner | 30 complete fresh-session A/B runs: five workloads, three repeats, 30 warm-up and 120 measured frames per run |
| Interpolation benchmark suites | 90 complete runs: seeded current rendering, seeded half-tick rendering, and fixed stepping compared with the earlier baseline |
| `cargo xtask asset import` | Original GLB imports deterministically; explicit reimport preserves placement and uses revision-checked saves |

Browser checks cover WebGPU initialization; 1,000, 10,000, and 50,000 cube workloads; one draw for the cube field and six mesh/texture draws for the game; paused-frame stability; pause across visibility transitions; resume; keyboard and pointer input; zero-sized suspension and restoration; narrow-viewport resize; no browser console errors; and unsupported WebGPU. Game checks exercise movement, collection, jumping, invalid replacement preservation, repeated revision changes, stable CPU/GPU asset residency, and zero repeated geometry/texture uploads after warm-up. The runner uses `--enable-unsafe-webgpu` for headless execution. Physical mobile hardware and real GPU device-loss recovery were not tested.

Desktop and narrow-screen screenshots were visually inspected and regenerate under `artifacts/`. Rust tests cover 15 asset tests, 64 core/input/stepping tests, 16 development/authoring tests, 18 rendering tests, 33 scene/schema tests, 6 game tests, 13 importer tests, and 15 tooling tests. They cover stale/foreign handles, texture/mesh budgets, action edges, exact stepping, snapshots, rollback/disposal, shader/UV layout, box landing, bounded inspection/serialization, legacy scene compatibility, schemas, guarded saves, malformed GLB/PNG/accessor/transform input, deterministic cooking, and explicit import replacement. Hierarchy regressions cover cycles, depth, generational parents, orphan isolation, composed shear, unchanged invalid-input caching, capacity recovery and safe extreme-scale normals against a double-precision reference. Culling tests include 5,000 deterministic cases checked against independent double-precision clip corners, all six planes, conservative extremes, budget/counter identities and retained-source behavior. Interpolation adds zero/multiple-tick, midpoint, shortest-arc rotation, hierarchy shear, discontinuity, replacement identity, invalid-midpoint fallback, 100,001-node overflow, storage reuse and exact agent observation checks. Node tests cover input, bridge, development lifecycle/sampling, streaming-source, source-watching, build-coordination, session replacement and benchmark report comparisons.

The agent browser loop verifies schema discovery, player inspection, exact 30-tick input movement, non-mutating invalid requests, screenshot/frame correspondence, invalid/stale save rejection, persisted candidate bytes, explicit reload of the accepted revision/name, assets, pause/resume, and clean diagnostics. Tests remove their own disposable sources afterward. Captures under `artifacts/agent/` were visually inspected. Separate checks confirm production ignores `engine_dev` and the scene selector, while Orbit supports camera inspection without an authored scene. Runtime replacement and main-frame navigation invalidate the old bridge session.

After adding managed binding selection, the full production browser smoke passes again. Both production games ignore a supplied `engine_build` UUID and `engine_dev=1`, expose no development API, and request no development or generation-specific bindings.

Automatic Rust iteration is verified with actual source edits and native Wasm builds. A deliberate compiler error returns file/line/column diagnostics while the active game's position, tick, and session remain intact and usable. Repairing the source with a changed movement speed creates a new build/session; six fixed steps move the player approximately 0.325 units instead of 0.65. Old session IDs are rejected, captures identify the new build, and browser diagnostics remain clean. The smoke restores its exact original source bytes, closes its watcher/browser/compiler ownership, releases both ports, and removes its lock. Unit regressions cover concurrent save retries and recovery from a persistent watcher error to identical source bytes. A successful Rust replacement starts a fresh game; it does not preserve simulation state.

The textured sample uses one 64×64 atlas (16,384 bytes) and two imported meshes shared by two scenery placements. Chrome reports six GPU meshes / 8,264 mesh bytes, one GPU texture / 16,384 texture bytes, and 25,992 CPU asset payload bytes including the compatibility cube. Warm mesh and texture uploads are zero. Invalid base64 and nonopaque texels preserve runtime state. Editing only texture pixels changed 4,362 screenshot pixels; equivalent repeated reloads differed at three edge pixels out of 695,156, within the test's explicitly bounded tolerance. Removing texture references and data retired all CPU/GPU textures. The native input fixture is original procedural art; its reproducible generator and GLB remain in `assets/source`, separate from scene placement.

The hierarchy sample expands two lantern placement roots into four rendered
children, plus the player's golden accessory: 33 entities and 31 renderables at
startup. Six exact ticks move the player and accessory by the same approximately
0.65 units while the accessory's local transform remains unchanged. A routed
nonuniform/rotated hierarchy verifies all 16 world-matrix elements, including
shear, without moving the other instance. A third placement adds three entities
and two renderables while keeping CPU/GPU asset counts and bytes and six draw
calls unchanged. Repeated replacement preserves residency; removing both garden
placements retires their GPU meshes/texture while declared CPU assets remain
scene-owned. Cycles, missing parents and expanded-ID collisions reject without
changing frame, revision, gameplay or residency. Browser captures are stored in
`artifacts/hierarchy-smoke-*/`; these tests route candidates without editing source.

Release Wasm sizes reported by Vite are approximately 1.34 MB / 359 KB gzip for Orbit and 1.74 MB / 478 KB gzip for Crystal Garden. These are artifact-size observations, not download-speed guarantees. The browser benchmark suite now records controlled local CPU phase costs and paced frame distributions; no GPU timestamp or cross-device performance claim is made. See [benchmarking](benchmarking.md) for configuration and repeat variation.

## Interpolation browser verification

`artifacts/interpolation-browser-checks.log` records the final interpolation,
hierarchy and culling runs against rebuilt development bindings. The ordinary
browser frame entry advances to tick one at 16.6667 ms and displays the halfway
pose at 25 ms: authoritative player Z is approximately -0.10833334, presented Z
is -0.05416667, with its child inheriting the same motion and no second tick.
Five explicit fractions preserve authoritative state. Half/current captures
differ by 784 pixels, establishing that the preview changes the actual image.

At a camera far plane, fractions 0, 0.5 and 1 submit respectively two, one and
zero objects; culling enabled/disabled images differ at zero pixels in each case.
Exact step/default capture, invalid fraction rejection, teleport/focus/suspension
resets, scene replacement, stale reload preservation and extraction identity all
pass. The two-node fixture retains 256 history bytes and 320 scratch bytes;
presented-component payload drops from 160 to 80 bytes when its child is removed.
Repeated reloads retain stable capacities. These are scoped payload counters,
not total process memory. Captures are under `artifacts/interpolation-smoke-*/`.

## Local interpolation measurements

`artifacts/interpolation-current-render.json` and
`artifacts/interpolation-half-tick.json` use identical code/source fingerprints,
browser/environment metadata and one setup tick. Each has 30 fresh-page runs,
30 warm-up frames and 120 measured frames with three repetitions. The following
rows disable culling so both modes submit the same objects. Values are medians
of the three per-run CPU medians, not production frame-rate guarantees.

| Workload | Current rendering | Half-tick rendering |
| --- | ---: | ---: |
| 1,000 cubes | 0.40 ms | 0.70 ms |
| 10,000 cubes | 1.70 ms | 3.40 ms |
| 50,000 cubes | 9.60 ms | 18.60 ms |
| Crystal Garden | 0.30 ms | 0.30 ms |

At 50,000 cubes, per-repeat medians range 9.3–9.7 ms current versus
17.8–18.7 ms half-tick. Extraction medians are 5.90 versus 14.70 ms;
paced frame p95 is 16.90 versus 33.40 ms. The additional cost is substantial.
Retained interpolation payload is 12,800,000 history bytes plus 8,000,000 scratch
bytes and 4,000,000 presented-component bytes: at least 24.8 MB, excluding maps,
ECS capacity/metadata and allocator overhead. Crystal Garden's corresponding
payloads are 1,792 + 5,280 + 2,640 = 9,712 bytes. Capacity can exceed live node
counts after a workload replacement because bounded allocations are reused.

`artifacts/interpolation-step-benchmark.json` is compatible with the earlier
culling baseline and includes another 30 runs. At 50,000 cubes with culling off,
fixed-step CPU median increases from 14.60 to 24.40 ms (about 67%); simulation
from 9.00 to 17.00 ms and extraction from 2.10 to 4.70 ms. This mode renders current
poses, so it exposes history maintenance and shared presentation overhead rather
than fractional blending. Per-repeat total medians are 23.0–25.0 ms, versus the
earlier 13.2–15.3 ms. These measurements support opt-in use and identify bulk
history/presentation processing as further performance work. They do not support
a claim that smoothing thousands of objects is free. The complete comparison,
including culling-enabled rows, remains in the artifacts and
`artifacts/interpolation-measurement-summary.json`.

## Local culling measurements

`artifacts/culling-benchmark-baseline.json` records the complete 30-run suite with
Chrome 153.0.8010.50, Windows x64, Core Ultra 9 275HX, 1440×1000 viewport and DPR 1.
The browser's separate adapter probe reported Intel xe-lpg and no fallback flag;
that probe is not a claim about physical execution of the engine device. No
concurrent repository build/test workload was running during measurement.
Values below are medians of each repeat's median, in milliseconds; frame p95
stayed around 16.9–17.0 ms in both modes.

This baseline predates interpolation and is retained with its code fingerprints
for comparison; the values below describe that earlier implementation.

| Workload | CPU total off | CPU total on | Submitted off → on | Instance/uniform upload bytes off → on |
| --- | ---: | ---: | ---: | ---: |
| 1,000 cubes, all visible | 0.50 | 0.60 | 1,000 → 1,000 | 80,064 → 80,064 |
| 10,000 cubes, partial view | 2.50 | 2.80 | 10,000 → 6,538 | 800,064 → 523,104 |
| 50,000 cubes, partial view | 14.60 | 13.60 | 50,000 → 23,872 | 4,000,064 → 1,909,824 |
| 10,000 cubes, camera away | 2.80 | 2.80 | 10,000 → 0 | 800,064 → 64 |
| Crystal Garden | 0.30 | 0.30 | 31 → 31 | 2,544 → 2,544 |

The 50,000-cube case reduced submitted instances and per-frame upload bytes by
about 52%. Its per-repeat CPU medians ranged 13.2–15.3 ms disabled and
12.8–15.9 ms enabled, so the small total-CPU difference should not be treated as a
robust speedup. Culling adds CPU work when visibility saves little rendering work;
the 10,000 partial case was slower in total CPU time. Simulation and transform
propagation still run for every object. These observations justify retaining the
toggle and repeatable measurements, not a universal FPS improvement claim.

## Transform pipeline optimization (25 September 2026)

`cargo xtask lint`, `test` (290 Rust tests, 91 Node checks), `build` and `docs`
pass after the interior culling test, SIMD validator path, contiguous stamp
scan, packed presentation matrices and retained pose view. Culling,
interpolation, hierarchy, collision and main browser smokes pass against fresh
development bindings; culling on/off captures differ at zero pixels. Matched
headless Chrome Stress Lab captures (90,000 all-moving renderables, three
interleaved fresh-page repeats per build) reduce one fixed tick plus one frame's
CPU from 32.2 to 19.8 ms raw, about 30% after normalizing by unchanged gameplay
time, with identical submitted counts. See
`artifacts/transform-pipeline-optimization-2026-09-25.md` for per-stage results,
the rejected packed-evaluation experiment and remaining costs. These are local
development measurements, not a Firefox or production frame-rate claim.

## Transform pipeline rounds 2–3 (25 September 2026)

`cargo xtask lint`, `test` (291 Rust tests, 91 Node checks), `build` and `docs`
pass after splitting hierarchy, presentation, history and retained-slot storage
by access pattern, writing retained pose rows during extraction, inlining the
validator fast path, revision-based collision sync and mapped staging for large
pose uploads. Culling, interpolation, hierarchy, collision and main browser
smokes pass against fresh development bindings. At the same tick, round 3's
exact and fractional Stress Lab captures are byte-identical to round 2's, with
identical submitted/culled/pose-update counts. The Stress Lab scene is unchanged
from the baseline. Three builds interleaved over six repeats in one headless
Chrome session (a loaded machine) reduce one fixed tick plus one frame by a
paired median of 30% against checkpoint 4 (34% excluding unchanged gameplay
time), from 24.1 to 16.6 ms, with presentation −29%, evaluation −39%,
extraction −31% and renderer packing −65%. See
`artifacts/transform-pipeline-round3-2026-09-25.md`. Staging targets Firefox's
`writeBuffer` cost and awaits a Firefox capture; these are local development
measurements, not a production frame-rate claim.

## GPU presentation (25 September 2026)

`cargo xtask lint`, `test` (304 Rust tests, 91 Node checks), `build` and `docs`
pass with GPU presentation ([decision 014](adr-014-gpu-presentation-and-culling.md)).
The only rustdoc warning (`[0,1]` in `engine_collision`) predates this work.
Main, culling, interpolation, hierarchy, collision, texture and the new GPU
presentation smokes pass against fresh development bindings. Native tests mirror
gathered inputs against dense and sparse CPU presentation, gate gathering on
changed inputs, check on-demand preparation and its refusal after a tick, list
invalid candidates and verify inspection materialization; shader tests validate
WGSL with naga, reject out-of-range decimal float literals that Chrome's compiler
refuses, and compare culling/admission constants and record layouts with core.
The interpolation smoke timed out on one command in three of its first nine runs
right after a rebuild (no browser errors; the command never returned within 10
seconds), then passed 20 consecutive runs and the final suite; the cause was not
identified. Cold compute-pipeline compilation took about 0.5 s in a fresh Chrome
GPU process, asynchronously.

At 90,000 all-moving Stress Lab objects in the user's configuration, GPU and CPU
captures of four fractions differ in 22–36 of 460,000 pixels (1–3 beyond a
tolerance of 8) and GPU visible counts equal the CPU culler's 89,080. Uncapped
headless Chrome throughput in one profiling build, ABBA over four repeats:
1,284 versus 82 fps on the forced NVIDIA adapter (paired ×16.1) and 276 versus
87 fps on Chrome's default Intel adapter (paired ×3.15), both at 60 fixed ticks
per second. Frames without a tick cost 0.1–0.2 ms of CPU on the NVIDIA adapter;
frames with a tick cost 10.6–11.6 ms at 90,000 objects and 7.2–8.2 ms at 60,000.
See `artifacts/gpu-presentation-2026-09-25.md`. These are local headless
measurements, not a Firefox or production frame-rate claim.

Wasm normal-dependency inspection confirms the importer, glTF parser, PNG decoder, source hashing, schema generation, and Bevy reflection are absent from the game's runtime dependencies. Production also excludes `engine_dev`; development schemas are generated by the host build. Production bindings contain no development exports and the bundle contains no development adapter chunk. Prior foundational dependency inspection found no duplicate Bevy, glam, or wgpu versions. The GitHub workflow uses these local commands but has not run remotely because this workspace has no Git repository/remote.

## Firefox 120 Hz work (25 September 2026)

`cargo xtask lint`, `test`, `build` and `docs` pass with pipelined fixed ticks
([decision 015](adr-015-pipelined-fixed-ticks.md)) and the leaner GPU inputs,
matrix admission and collision refit described in
[decision 014](adr-014-gpu-presentation-and-culling.md); the only rustdoc warning
predates this work. The main, GPU presentation, interpolation, hierarchy,
collision, culling and texture smokes pass against fresh development bindings.
The interpolation smoke's ordinary-frame check now disables pipelining, because
its development commands between frames would complete a started tick early.
New native tests cover bit-identical `Transform::matrix` lanes (200,000 seeded
and special values against glam), the vector admission gate against the original
predicate, refitted collision trees against trees built from scratch, pipelined
ticks against unpipelined ones at 120 Hz, early ticks hidden from GPU frames
until due or revealed, and published stamp counts as upper bounds in the
simulated consumer. Debug builds assert the cached scale-compatibility bit
against its direct computation at every use.

In Firefox 156 at 120 Hz, main-thread work per fixed tick of the user's
90,000-object all-moving workload fell from 15.7–16.0 ms to 13.5–13.9 ms and
frame rates rose from 49 fps to 77–89 fps in the final comparison session
(page-to-page variation is about ±15–20%). See
`artifacts/firefox-120hz-2026-09-25.md`. These are local headful measurements,
not a production frame-rate claim.
