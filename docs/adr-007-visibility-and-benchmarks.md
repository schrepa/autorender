# Decision 007: conservative visibility and measured browser workloads

Status: adopted. Extends decisions 002, 003 and 006. Core owns visibility tests;
assets own immutable local bounds; the renderer owns GPU residency; optional
development tooling owns benchmark sampling. No visibility logic runs in JS and
no GPU objects enter gameplay systems.

## Visibility and ownership

`Mesh::new` computes finite local `Aabb` min/max endpoints once while validating
vertices. Bounds include all vertices, including unused vertices, and are metadata
outside the reported vertex/index payload bytes. No per-frame vertex scan occurs.

Frustum culling is enabled by default in `EngineConfig.frustum_culling`.
`Engine::set_frustum_culling` allows controlled comparisons and custom clients to
disable it. Extraction derives six inward planes from exactly the camera matrix
sent to rendering, including WebGPU's near plane at clip depth zero. It transforms
each plane into mesh-local space and evaluates a support point of the local box.
The full world matrix preserves hierarchy shear, rotation, nonuniform scale and
reflection. Intersection and numerical uncertainty keep the object visible.

Floating-point comparisons use a margin scaled by sums of absolute products.
Extreme coefficients, overflow risk and nonfinite intermediates disable rejection
for the affected test. This conservative fallback can retain extra objects; it
must not clip potentially visible triangles. Core tests compare decisions with an
independent double-precision clip-corner reference. No spatial index, occlusion
culling or GPU culling is implied. Work remains linear in renderable entities.

Most boxes in a wide view lie wholly inside it, where the support test must
evaluate all six transformed planes before keeping them. A cheaper interior test
runs first: two products build an axis-aligned world box around the transformed
local box, and four-lane plane arithmetic proves every point strictly inside all
six planes. Its rounding proof (beside the implementation) needs a doubled
center/extent, coefficient/product guards and a relative margin; non-affine,
extreme or unproven boxes take the support test. Acceptance proves visibility
while support rejection proves exclusion, so the classification always equals
the support test alone. It only removes arithmetic; tests check that equivalence
and a double-precision all-corners-inside reference.

Counters partition each extraction:

```text
render_candidates = invalid_transforms + missing_assets + culled_instances + visible_instances
visible_instances = snapshot.instances.len() + dropped_instances
```

Invalid transforms and missing dependencies are checked before visibility.
`visible_instances` counts survivors before the extraction instance budget, so
culled objects consume no instance slots. Disabling culling produces zero culled
instances and provides the comparison baseline.

Snapshot source lists retain assets referenced by emitted **or culled** valid
renderables. The renderer uploads only sources required by emitted draws, while
keeping already resident GPU resources alive if their source remains in the
snapshot. An object first seen off-screen causes no upload. Moving out and back
does not evict and reupload its assets. Removing its last valid renderable retires
GPU resources at the next valid boundary, including when all draws are absent.
Missing/invalid/budget-dropped objects alone do not retain source resources.
Existing source-count and byte budgets apply before any GPU mutation.

Static-work reuse keeps these boundaries intact. Core caches conservative
visibility results for a bounded prefix of 262,144 (formerly 100,000) valid inputs, comparing exact
model bits and immutable mesh handles under an unchanged camera/culling mode.
Live asset resolution still precedes reuse; snapshots still own their outputs.
The cache retains matrix/weak-handle records, not another copy of scene assets.
Beyond the cache bound, inputs take the ordinary conservative test.

The renderer retains grouped instance indices and draw ranges when the exact
input-order mesh/texture keys match. It validates source lists on every call and
refreshes source mappings even on reuse. Invalid snapshots invalidate the plan
before returning. Rebuilding uses bounded counting scatter in mesh/texture order,
preserving source order within each draw group so moving/static runs stay coherent.
Camera and packed instance bytes are compared with previously queued data;
unchanged records skip repacking and uploads. Changed records produce at most 32
coalesced writes. Fragmented/dense plans use one full write; changed counts,
replaced buffers and uncommitted prior preparations also require a full write.
All selected writes must reach the queue before reuse is committed. The existing
staging vector is the only retained instance-data copy. See the [renderer README](../crates/engine_render/README.md)
for byte/call heuristics and scratch-memory bounds.

This targets the measured static-scene costs of repeated culling, sorting and
uploads. Input comparison, live asset checks, snapshot copying and named ECS
observation boundaries remain. Additional bounded cache memory trades against
repeated computation; profiling reports cache capacity, reuse and actual writes.
Timing improvements require matched browser measurements.

This changes the low-level snapshot contract: source lists may contain zero-draw
resources. They do not request eager uploads. A snapshot retains immutable CPU
sources independently of scene disposal, as before. Visibility never despawns
entities, changes simulation or alters scene asset ownership.

## Agent observations and controls

Browser stats and development replies expose candidates, visible, culled, emitted,
dropped, missing and invalid counts, plus the current culling mode and upload
bytes. `Mesh3d` inspection exposes local bounds. Counters describe the latest
render extraction; they are not a persistent visible-entity query.

Paused `culling` and `camera` controls render the resulting state without advancing
clocks. Camera controls validate the complete numeric pose/lens before mutation
and align an existing fly controller while preserving speed/sensitivity. That
controller retains its documented pitch clamp. A subsequent render error reports
that the temporary state may already have changed. These commands do not save
source or change authored camera fields.

The host's optional `benchmark` command accepts render-only or one-fixed-tick
sampling, 0–120 warm-up frames and 1–240 measured frames, with a ten-second
deadline. It requires a paused visible nonzero canvas and holds the host's busy
boundary through every await. Pending input is cleared. Step mode advances
gameplay through warm-up and measurement; render mode does not. Resize, suspension,
session disposal or deadline expiry aborts sampling without presenting an
incomplete result as a successful benchmark. Already completed steps remain real.

CPU phase times come from Rust's existing `performance.now` observations and
exclude command serialization/stat copying. Samples are paced by animation frames.
Each frame interval spans its measured operation to the following RAF boundary,
including the final sample. Those intervals include browser scheduling and host
work; they are not GPU execution times. All percentiles use linear interpolation
over actual samples. CPU asset and GPU mesh/texture byte counts exclude allocator,
instance-buffer, depth-buffer and full process/device memory.

`cargo xtask benchmark` and its direct Node runner create fresh paused sessions
for each fixed workload and culling mode, alternate A/B order, and write versioned
JSON. Reports retain per-run measurements, metadata, source/binary hashes and
repeat variation. Comparisons reject incompatible device/browser, workload or
sampling metadata. Opt-in regression thresholds provide nonzero exit status;
there is no universal frame-rate promise or default hardware-independent gate.
See [benchmarking](benchmarking.md) for commands and measured limitations.

Production excludes the sampling adapter and control schemas. Ordinary frames
add only native bounds testing/counters; no benchmark serialization or JS sampling
occurs. The existing HUD still polls aggregate diagnostics at low frequency.
