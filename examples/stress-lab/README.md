# Stress Lab runtime

This client generates seeded workloads through the public `engine` façade,
selected by `scene`: a tower field, a procedural city, the Metro (the city's
plan built with every engine feature) and dense virtual geometry; see
[`docs/stress-lab.md`](../../docs/stress-lab.md).
It owns its plugins, motion, workload configuration and diagnostics. No gameplay
or stress-specific code lives in an engine crate.

The browser's Apply action validates its configuration and builds a complete
replacement engine before swapping at a frame boundary. Mesh, material and
texture handles include their originating asset-library identity, so old
renderer snapshots cannot resolve against new assets. Replacing a workload
resets simulation, query totals and sensor history. Browser cumulative dropped
time remains a lifetime counter; compare deltas within a recording.

Workloads take 1–4,000,000 renderables. In the tower field, hierarchy groups add
one transform-only parent per 32 renderables. Every group uses a rotated, nonuniform parent, with
rotated children exercising composed affine transforms. Moving percentages
select deterministic indices; interpolation applies only to moving entities.
In the tower field, mesh variety chooses 1, 8, 32 or 64 unique mesh handles and
matching materials, each with an optional 64 × 64 opaque checker texture. A default built-in cube also
remains resident in the CPU asset library.

Up to 10,000 renderables receive independent sphere/cuboid colliders. Dense mode
packs those objects into a tight volume to deliberately challenge broad-phase,
narrow-phase and sensor-pair budgets. Every query error is counted; incomplete
sensor attempts remain visible and retain the engine's last complete pairs.
Dense mode is intentionally expensive even if few objects are visible.
When both collider and query counts are zero, the collision plugin and its
timing systems are absent. Disabled animation and query workloads also register
no fixed systems. Observations expose `collision.enabled: false` with zero
counts and no attempted sensor tick for these isolated rendering workloads.

Queries run after animation against the authoritative cache refreshed before
gameplay. Ray, overlap and translation-only shape-cast workloads cycle across
known collider positions. Overlaps use a reusable 128-result buffer; exceeding
that limit produces a counted error instead of partial hits. The configured
query count is per fixed tick, so catch-up frames may execute it five times.

`stress_stats()` produces small JSON observations only when polled. Collision
pre/post sync, sensor detection, animation and query timers are CPU wall-clock
subsets of the frame's simulation time. They must not be added to simulation
again, and they do not cover all transform propagation, history or scheduling
work. Rendering timings come from the runtime's frame statistics (`stats()`
JSON) and are CPU submission time; GPU execution time needs a profiling build
(`docs/profiling.md`). Reported memory values are bounded payload estimates,
not total process or GPU-driver memory.

Development discovery includes `StressConfig` and `StressMetrics` alongside
standard engine resources and spatial query operations. Configuration is procedural runtime
state, separate from authored scene files. The host provides the reviewable
configuration and measurement export; no benchmark starts automatically.
