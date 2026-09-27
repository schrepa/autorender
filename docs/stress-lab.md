# Stress Lab

Open `/stress.html` on the running server. The page starts with 90,000 textured
towers and 32 mesh batches. Select a preset to replace the workload, or adjust
individual controls and Apply (up to 250,000 objects; the engine admits 262,144
transform nodes and render instances). Failed configuration preserves the active
world.

| Preset | What it exercises |
| --- | --- |
| Maximum visual | 90,000 static objects, 32 meshes and textures; no collision plugin or motion systems |
| Mixed | 10,000 objects, moving hierarchies/interpolation, 1,000 colliders and 32 queries per tick |
| Transforms | 50,000 moving objects with parent hierarchies and interpolation |
| Queries | 10,000 colliders, moving bounds and 256 ray/overlap/sweep queries per fixed tick |
| Dense sensors | 2,500 sensor colliders in tight clusters; pair-budget exhaustion is deliberately observable |

Overview, inside-field and look-away cameras isolate visibility changes. Culling
and resolution controls isolate draw submission and pixel costs; mesh variety
changes actual draw batches. Resolution remains bounded by the shared host's
2560×1440 drawing-buffer budget. These settings do not imply GPU timestamps.

Click Record in the workload panel to capture while the scene stays visible,
then Export JSON. Camera input and workload controls lock during recording.
Pause or changing visibility aborts with an incomplete report. Copy workload link
preserves the configuration and seed and opens a fresh overview view. The
source of this scene is procedural Rust plus the explicit JSON configuration,
not the 10,000-entity authored-scene file format.

Agents can launch `cargo xtask agent watch --game stress`, discover `StressConfig`
and `StressMetrics`, and use the standard inspect/step/benchmark interfaces. A
`config` JSON query parameter selects a repeatable workload at startup. Ordinary
frames use GPU presentation ([decision 014](adr-014-gpu-presentation-and-culling.md));
`presentation=cpu` keeps the exact CPU presentation path for same-build
comparisons, and recordings and profiles report the selected `presentation`. Those
frames also pipeline fixed ticks ([decision 015](adr-015-pipelined-fixed-ticks.md));
`pipeline=off` runs each tick when due, and recordings report the setting.
`taa=on` temporally antialiases GPU-presented frames
([decision 017](adr-017-temporal-antialiasing.md)); recordings and profiles
report `taa`. Spatial
queries require the collision-enabled presets. No automated suite starts when
opening the page. Runtime code lives in `examples/stress-lab`, independent of
engine implementations.

The Stress Lab has an opt-in production recording helper. Recording observes
the running workload and exports JSON; it does not enable the development
protocol, pause gameplay, execute extra simulation steps or change the scene.

Keep the same seed, configuration, camera, culling setting and drawing resolution
when comparing recordings. The page includes these settings in the report's
metadata. The page records the startup Wasm URL and SHA-256 when available, plus
the managed build ID; absent fingerprints are explicit nulls. Never rebuild a
mutable development server during a run. Browser identity, viewport, device pixel ratio and available browser
hardware hints are captured once at recording start. An existing runtime's
adapter description may accompany the metadata; the helper never requests a
second WebGPU adapter or device. Browser hints do not establish physical GPU use.

A default recording discards 30 warm-up frames, then measures ten seconds of
animation-frame intervals. Its sample count depends on the browser's actual
cadence: a slow workload can produce a complete recording with fewer frames.
CPU work for each frame is paired with the **following** animation-frame
interval, including the last measured frame. Reports preserve the raw timing
samples and min/median/p95/p99/max distributions. Percentiles use linear
interpolation at `(sample_count - 1) Ã— fraction`.

CPU total is simulation plus extraction plus render preparation/submission.
Additional phase timers, such as collision work inside simulation, are subsets;
adding them again would double-count their cost. CPU stage timers exclude
recording/stat collection, while frame intervals include host work and browser
pacing. **GPU execution time is not measured.** These recordings also do not
measure JS/Wasm/process heaps or total driver/GPU memory.

Per-frame numeric counters retain their first/last values and measured minima
and maxima, exposing transient missing assets, invalid transforms, overflows and
other diagnostics even when the last frame has recovered. Boolean completeness
counters use 0/1 minima and maxima. Stress-specific JSON
snapshots are retained at most four times per second. Dropped simulation time is
the change in the cumulative counter over the measured frames; with zero warm-up,
the first observed frame has no preceding counter baseline.

Configuration, camera, resize, visibility and pause changes must abort an active
recording through the page's lifecycle hooks. An interrupted report remains
exportable with `complete: false` and its reason. A completed report means the
time window finished; inspect diagnostic counters before treating a workload as
healthy or comparable. There is a 30-second wall-time deadline including warm-up
and a cap of 1,800 measured samples. Reaching either limit before ten measured
seconds produces an incomplete report. Metadata is capped at 16 KiB; individual
diagnostic snapshots at 8 KiB, with at most 121 snapshots per recording.

Close unrelated browser workloads and avoid builds during comparisons. Repeat
matched runs and retain each JSON artifact: browser scheduling, power state and
background work can change results. Use the separate [benchmark runner](benchmarking.md)
for automated fresh-page repetitions with compatibility checks.

The page owns UI, export and runtime lifecycle. `web/src/stress-recording.ts`
provides `StressRecorder.start(metadata, options)`, `sample(frame)`,
`diagnostic(timestamp, snapshot)` and `abort(reason)`. `active` allows the host to
read per-frame Wasm stats only during recording; ordinary rendering retains its
existing low-frequency HUD polling. Stats bindings remain owned by the host and
must still be released after copying their fields.
