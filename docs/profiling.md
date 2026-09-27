# Optional engine profiling

Detailed profiling is a separate build of Stress Lab. Ordinary game builds keep
inline no-op hooks and include no collector, allocator wrapper, timestamp queries
or profiling host module. Disabled `engine_profile` has no external dependencies.
Capturing adds overhead: verify improvements with ordinary recordings and
[matched browser benchmarks](benchmarking.md).

## Run and capture

Activate the workspace tools, then build and serve:

```sh
cargo xtask profile
npm run preview --prefix web -- --outDir dist-profile --port 5175 --strictPort
```

Open **http://127.0.0.1:5175/stress.html**, choose the workload, let it settle,
and select **Profile 5s** beside Record to keep the scene in view, then export
from the detailed profiling panel below. The JSON contains raw
measurements and `traceEvents` for Perfetto or Chrome tracing. Controls lock
during capture and GPU settlement. An early stop, interruption, overflow or failed
supported measurement is marked incomplete. Unsupported capabilities are reported
separately; a complete CPU capture does not imply GPU timestamps were available.

`cargo xtask profile` generates `web/pkg-profile` and `web/dist-profile`, with
release optimization and retained debug information for function sampling. The
first local Wasm artifact was about 28 MB versus about 1.74 MB for ordinary Stress
Lab; these are uncompressed artifact sizes, not performance measurements. The
production bundle and existing game at port 5174 remain usable. Rust edits require
another profile build and page reload; managed watch builds retain their ordinary
development configuration.

## Coverage

| Area | Captured detail |
| --- | --- |
| Core CPU | Browser frame/input, fixed ticks and named schedules, hierarchy input scans/topology rebuilds/dirty propagation, interpolation history/presentation, combined extraction scan/assets/culling/copy, instance counters |
| Collision | Synchronization/cache and BVH rebuilds, ray/overlap/shape-cast spans, sensor detection, aggregated BVH traversal and precise-test time/counts |
| Render CPU | Source validation/indexing, stable instance grouping/draw ranges, residency retirement, surface acquisition, asset uploads, packing, queue writes, encoding and submission |
| GPU | Asynchronous timestamps for the opaque pass and, on GPU-presented frames, the presentation compute passes and each presentation stage when `timestamp-query` is supported; capture/frame/submission IDs accompany each duration |
| Memory | Rust allocation/deallocation/reallocation counts, requested/freed/live bytes and lifetime peak; per-frame trends, extraction/batch/staging capacities, GPU instance-buffer and resident asset payload bytes; Wasm capacity and optional JS heap samples |
| Browser | RAF cadence and callback durations, Wasm/host boundaries, supported long-task and long-animation-frame entries with bounded attribution; optional agent CPU sampling |

There are no per-entity trace records, allocation stacks, automatic wrappers around
arbitrary ECS systems, shader invocation statistics or portable GC events. Add a
coarse gameplay scope where needed:

```rust
let _profile = engine::profiling::span("game", "enemy_navigation");
```

Function-level investigation uses the agent's sampled CPU profile or browser
DevTools. Sampling is statistical; optimized/inlined functions may not appear as
distinct frames (see [function attribution](#function-attribution)). Collision aggregates sum operation timings into one event: they
are counters, not continuous timeline spans. Nested CPU scopes and the host
callback overlap and must not be added together.

Hierarchy counters describe each propagation boundary: `hierarchy.topology_rebuilt`
is 0 or 1, `hierarchy.evaluated_nodes` counts derived-matrix evaluations, and
`hierarchy.written_globals` counts changed canonical global outputs. Stable local movement
should leave topology rebuilds at zero; stationary boundaries should evaluate and
write no globals. `change_tick_scan` applies an engine-owned change window to
each table's contiguous `Transform` stamps: it skips a table whose stamps are all
unchanged in one unrolled pass and otherwise classifies 16-stamp blocks before
fetching payloads; the decision equals Bevy's `Changed<Transform>` test. It still
visits metadata on unchanged boundaries, but only changed rows compare payloads,
and they load their source record only for a re-added `Transform`. Earlier captures used the row-by-row `Changed` filter, so
compare unchanged-boundary durations across that change with care. The source loop also captures interpolation endpoint changes,
so its duration includes that work. No per-entity timers are added to separate
those operations. Successful propagation acknowledges the source scan before
advancing the ECS tick; no derived-component writes need acknowledgement.
`apply_transform_edits` measures deferred replacements only when a caller uses
that API; direct mutable-query gameplay needs no edit buffer or apply phase.

`evaluate_dirty` evaluates changed branches and publishes canonical matrices.
Decision 013 removes the separate `write_globals` ECS publication span and the
`hierarchy.dense_writes` column-path decision. Earlier captures contain these
scopes/counters and, before decision 011, `flush_transform_observers` as well.
Explicit command queues flush before propagation; the complete `propagate` scope
includes that boundary work.

`hierarchy.metadata_nodes` counts matching local-transform rows for timestamp examination.
On the ordinary path, `hierarchy.validated_nodes` counts rows admitted by the
local change filter. On a completed forced scan, it counts admitted
transformed rows checked without stamp filtering. `hierarchy.full_scan` indicates
forced validation at startup, structural changes, interrupted propagation, aged
tick windows or capacity admission; `hierarchy.scanned_nodes` counts transformed
rows in that scan. Ordinary stamp filtering is independent of
`World::clear_trackers`. Changed rows resolve cache indices by full Entity identity;
unrelated archetype moves do not require topology reconstruction. Early detection
of incompatible source order can instead require the separately measured rebuild.

`hierarchy.global_matrix_checks` remains zero as a compatibility counter; globals
are private canonical values and there is no external derived-component repair.
Previous
captures' `queued_edits`, `dense_scan`, `journal_overflow`, `tracked_validation`
and `memory.transform_journal_bytes` describe the superseded replacement journal
and are no longer emitted. Compare full simulation/propagation totals across
this architecture change rather than equating individual renamed scopes.

`transform_edits.queued`, `.applied` and `.skipped` remain available for the
optional deferred replacement API. Targets missing at admission are skipped;
capacity rejection happens while queueing and never drops accepted work.
`transform_edits.buffer_capacity_bytes` is one system's retained buffer, not a
sum across systems. `memory.global_write_buffer_bytes` is now zero because
canonical output eliminates the separate publication buffer. Older captures
counted retained changed-node indices or payload/suppression batches there.

`capture_tick_begin` advances the interpolation epoch; it performs no history
query. `capture_tick_end` closes the interval and consumes only coalesced validity
transitions, normally an empty list. Source observations already occurred inside
hierarchy processing. `update_history_in_place` and `history_index_rebuilt` now
belong to rare structural membership reconciliation, including nested
`rebuild_history_index` work; do not add the nested scope twice. History bytes
include the cold identity records, the hot 64-byte intervals (previous endpoint,
epoch and validity) and each record's node index, for both reusable buffers.
Current poses come from the hierarchy's local-pose array, avoiding a duplicate
transform per history. Source callbacks and epoch bookkeeping replace repeated
full capture passes; ordinary motion updates only the hot interval.

`prepare_blended_branches` measures display preparation;
`interpolation.evaluated_nodes` counts visited nodes including inherited/fallback
branches and writes directly into canonical presented storage. The former
`apply_presented_values` ECS output/repair pass is removed.
`interpolation.written_presented` counts changed canonical outputs where emitted.
Exact-current output uses `prepare_current_values`; no component repair follows.
All scopes nest inside presentation.
`interpolation.dense_preparation` selects direct parent-first blending when
history count reaches 50% of cached nodes; smaller populations keep sparse branch
traversal. This crossover is provisional. `history_index_reused` counts histories
whose cached hierarchy index still matches the current source. Dense preparation
also consumes source-to-history hints directly in that parent-first pass; it no
longer performs a separate history scatter or touched-slot reset pass. See
[decision 012](adr-012-transform-validation-and-presentation.md) for compact
scratch storage, sufficient matrix validation and small-angle rotation blending.
`interpolation.blend_blocks` counts dense four-pose numeric blocks. Their SIMD
lanes use bounded stack arrays; they do not add a retained full-scene pose store
or contribute a new Vec capacity. World composition remains parent-first.

`PresentationState.presented_store_bytes` reports canonical displayed-pose/entity
Vec capacity (matrices, revisions, traversal flags and entities), while
`presentation_scratch_bytes` reports temporary traversal/blend storage. `presented_component_bytes` remains zero for compatibility. Compare the
combined relevant capacities across the storage change, not the component field
alone; these are bounded payload observations, not process-memory totals.

The retained browser path uses `extraction_retained_updates_culling`.
`extraction.retained_resolved` counts source metadata resolutions;
`extraction.retained_pose_updates` and `.retained_color_updates` count changed
per-slot pose/color stamps. A changed pose also rewrites that slot's 48-byte
`RenderFrame::pose_rows` entry from the matrix extraction already read, so no
renderer pass re-reads the presented store. `memory.retained_render_capacity_bytes`
reports slot owner/frame-state, visit-mask, query-hint, visibility-index,
free-slot, change-stamp and pose-row Vec capacities; it excludes HashMap/source-pool
and allocator overhead. `extraction.instance_capacity_bytes` reports the retained
`RenderInstance` Vec capacity, including inactive slots, not visible instance
count. Public owned snapshots remain available; distinguish their extraction
path from retained frames when comparing work. See
[decision 013](adr-013-canonical-poses-and-retained-rendering.md).

On the retained path, GPU storage uses core slot indices: each affine pose is
48 bytes and each color is 16 bytes. Grouped 4-byte slot IDs select submitted
instances without moving those records. `render.pose_write_bytes`,
`render.color_write_bytes` and `render.instance_id_write_bytes` separate the
actual queue writes; `render.instance_write_bytes` includes all three streams.
`render.colors_repacked` counts color records rewritten in staging;
`render.poses_repacked` counts pose slots selected for upload, whose 48-byte rows
extraction already wrote into `RenderFrame::pose_rows` (the renderer no longer
repacks or stores poses). Range coalescing may upload unchanged neighboring
records. Pose uploads of at least 256 KiB go through persistently mapped staging
buffers: `render.pose_staged` is 1 when one was used, and
`render.pose_staging_unavailable` is 1 when none was mapped yet and the upload
fell back to `writeBuffer`. `memory.gpu_pose_staging_bytes` reports the staging
ring's capacity (at most three buffers). Queue-write time moves between browsers
with this path: Firefox copies large `writeBuffer` calls through new shared
memory, while Chrome's `writeBuffer` is already cheap.
`render.retained_frame` distinguishes this path from public snapshots' validated
80-byte instance records. Visibility or metadata changes can rebuild grouping
and the ID stream without rewriting unchanged colors. Slot capacity, including
inactive entries, can exceed the submitted count.

GPU-presented frames ([decision 014](adr-014-gpu-presentation-and-culling.md))
replace `prepare_blended_branches` with `prepare_gpu_presentation`. Its
`gather_gpu_inputs` scope runs only when a tick, refresh, reset or topology change
altered the inputs; `interpolation.gpu_gathered` is 1 then and 0 when the last
gather is reused. `interpolation.gpu_current_changed`, `.gpu_previous_changed`,
`.gpu_info_changed` and `.gpu_authority_changed` count records that gather
stamped (current poses are the hierarchy's cached locals, never copied), and
`interpolation.gpu_shift` is 1 when consumers copy current poses into previous
ones on the GPU instead of receiving explicit previous poses;
`interpolation.gpu_parents` counts active parents composed before candidates,
`interpolation.gpu_visited` counts the nodes the gather visited (every node for a
full walk, only nodes whose inputs can differ otherwise), and
`memory.gpu_input_bytes` reports the gathered CPU arrays.
`extraction.gpu_structure_reused` is 1 when unchanged renderables skipped the
per-renderable pass. The renderer reports `render.gpu_presentation`,
`render.gpu_candidates`, `render.gpu_node_upload_bytes`,
`render.gpu_group_upload_bytes`, `render.gpu_full_node_upload`,
`render.gpu_node_upload_staged`, `render.gpu_parent_levels`, `render.gpu_shift`
(the copy ran before this frame's uploads), `render.gpu_blend` (0 while a missed
copy suspends blending), `render.gpu_current_prestaged` (the frame copied current
poses staged by the preceding frame, see below) and `render.gpu_visible_readback`,
the latest completed visible count. On these frames
`render.instances` is that count within the budget and lags by a few frames, and
pose and slot-ID write counters stay zero. `memory.gpu_presentation_bytes` covers
the pass's GPU buffers. GPU samples named `presentation_compute` time the frame's
compute passes separately from `opaque_meshes`; count both as GPU work. During a
capture each presentation stage runs as its own timed pass and reports
`presentation_compose`, `presentation_cull`, `presentation_scan` (group sums and
draw arguments) and `presentation_scatter` samples; `presentation_compute` then
spans the first to the last of them. `presentation_shift` times the per-tick copy
of previous poses, submitted before the frame's uploads and not included in
`presentation_compute`. Splitting the stages adds pass boundaries, so compare
per-stage samples within captures rather than with uncaptured frames.

Temporally antialiased frames ([decision 017](adr-017-temporal-antialiasing.md))
add a `taa_resolve` GPU sample for the full-screen resolve; their
`opaque_meshes` also writes the motion target. They report `render.taa`,
`render.taa_history` (0 when the history was reset), `render.taa_object_motion`
(0 on frames that fall back to camera motion) and `memory.gpu_temporal_bytes`
(targets and the second pose buffer). The resolve scales with canvas pixels, so
compare it at identical sizes.

Textures report `render.texture_mip_levels`, the mip levels the frame rendered
for textures that became resident (their base levels still count in
`render.texture_upload_bytes`), and `memory.gpu_texture_mip_bytes`, the GPU
bytes of all generated levels beyond `memory.gpu_texture_payload_bytes`.

With pipelined ticks ([decision 015](adr-015-pipelined-fixed-ticks.md)) a tick's
stages appear as `fixed_tick_begin` and `fixed_tick_end` spans, usually in
different frames, instead of one `fixed_tick`; frames that present an in-flight
or early tick record `present_gpu_fraction` instead of a gather. Count ticks as
`fixed_tick` plus `fixed_tick_end` spans. A frame that completes a tick ahead
also gathers its GPU inputs (`gather_ahead`, containing `gather_gpu_inputs`)
and copies its current poses into a mapped staging buffer (`prestage_current`),
so the tick's due frame gathers nothing (`interpolation.gpu_gathered` 0) and
records only a GPU copy for those poses: its `queue_writes` excludes them, and
`render.gpu_node_upload_bytes` still counts them. `render.gpu_held` is 1 on a
frame that presented while such a gather waited for its due frame.
`hierarchy.deferred_globals` counts leaves whose world matrices were certified
instead of composed ([decision 016](adr-016-deferred-leaf-globals.md)); they
are included in `hierarchy.evaluated_nodes`. Collision reports `bvh_refit` when
moved colliders refit the tree and `bvh_rebuild` when membership changed or a
refit degraded it.

Uncapped GPU-presented frames can run thousands of times per second and exhaust
the 65,536-event budget within a second: check dropped events, use shorter
captures, and measure throughput from engine frame/tick counters without a
capture. Report frames that ran a fixed tick separately from frames that did not;
their costs differ by an order of magnitude.

Static rendering exposes `extraction.culling_tests` / `extraction.culling_reused`
and `render.batch_order_rebuilt` / `render.batch_order_reused`. The latter count
decisions per frame, not instances. `extraction.culling_interior` counts tests
accepted by the cheap interior proof; the remaining tests ran the transformed-plane
support test. `render.instances_repacked` counts staging
records rewritten; `render.camera_write_bytes` and `render.instance_write_bytes`
report actual queue writes. `render.instance_write_ranges` counts instance queue
calls. Public snapshots allow at most 32 ranges; retained frames allow at most
32 pose ranges, 32 color ranges and one ID-stream write.
`render.instance_full_upload` identifies full channel writes, and
`render.instance_upload_fallback` identifies fragmentation/density/call-cost
fallbacks. Fresh buffers, changed stored-slot counts and uncommitted retries force
full retained channel writes without setting that fallback counter. Changing
only visible membership does not change the stored-slot count. Group rebuilds
use `batch_instance_group`
instead of the old `batch_instance_sort` scope; source order is stable within each
mesh/texture group. An unchanged rendered frame can report zero upload bytes
while continuing to issue draws. Visibility and batch cache capacities appear in
memory counters; upload reuse retains the existing staging vector.
`memory.render_upload_range_bytes` is included in `memory.render_upload_cache_bytes`,
and `memory.batch_group_scratch_bytes` is included in `memory.batch_plan_capacity_bytes`.
These counters distinguish work avoided from work merely moved into input scans.

## Function attribution

Release code inlines the per-node helpers into a few large functions, such as
`GlobalTransforms::propagate` and the GPU input gather, so samplers report each as
one frame. Firefox attributes Wasm samples to whole functions, not instructions.
An attribution build keeps selected helpers out of line:

```sh
cargo xtask profile --attribution
npm run preview --prefix web -- --outDir dist-attribution --port 5176 --strictPort
```

It is the profiling build plus the `attribution` feature, written to
`web/pkg-attribution` and `web/dist-attribution`. The feature replaces the inline
attributes of the hierarchy's change-stamp block test, leaf certification,
deferral, derived-global composition and commit, deferred-global reads, local
matrix construction, global admission and certificates; interpolation
motion/source/validity observation; and collision's pair test, box separating-axis
test and GJK intersection with `#[inline(never)]`. Compare shares, not absolute
times: calls add overhead and change the surrounding code. At 90,000 moving objects
in Firefox, propagation measured 3.94 ms per tick split and 4.05 ms inlined under
the same conditions. Confirm every change on the ordinary build. To split another
helper, give it the same pair of attributes:

```rust
#[cfg_attr(not(feature = "attribution"), inline(always))]
#[cfg_attr(feature = "attribution", inline(never))]
```

Leave helpers of a few instructions inline: per-node call overhead would exceed
what they measure.

## Native timing aid

For quick comparisons while editing core transform/extraction code, an ignored
Stress Lab test builds the 90,000-object all-moving workload natively and prints
per-stage span minima for two ticks per displayed frame:

```sh
cargo test -p stress-lab --release --features profiling --test pipeline_timing -- --ignored --nocapture
```

Native x86 code generation, SIMD backend and memory behavior differ from browser
Wasm. Alternate builds, pin the process to performance cores on hybrid CPUs, and
confirm any result with matched browser captures before claiming it.

## Ownership, bounds and interpretation

One capture runs on the current thread's collector. The browser composition
supplies `performance.now`; neither core nor the collector imports browser/GPU
APIs. The renderer owns GPU resources and late results. The host owns observers,
recording controls and trace conversion; the local bridge owns artifacts and CDP.

| Bound | Value |
| --- | --- |
| Default / permitted duration | 5 seconds / 100–5,000 ms |
| CPU events | 65,536 default; browser/agent accept 256–65,536 |
| GPU storage | 3 reusable asynchronous slots; 4,096 samples per capture |
| GPU settlement | Up to 250 ms after CPU stop; no synchronous GPU wait |
| Browser storage | 4,096 frames, 1,024 observer entries, 64 memory samples |
| Deadline | 15 seconds, enforced when the browser thread next yields |
| Agent artifact size | 32 MiB per JSON or CPU-profile file |

CPU events allocate their budget before collection. The first capture includes
lazy GPU profiling setup in its allocation delta. Buffers retain bounded capacity;
overflow is counted instead of silently growing. A new capture replaces the
retained report. GPU results from an earlier capture cannot enter the new one.
Inspect completeness, capabilities, dropped counts, pending results and errors.

GPU durations exclude uploads, queue latency, presentation and timestamp readback.
Their clock is not calibrated against the CPU clock. Trace GPU counters are placed
at the originating CPU frame only for attribution, not as GPU execution spans.
Browser timestamp precision may be reduced, including valid zero-duration samples.
The renderer uses wgpu's [pass timestamp writes](https://docs.rs/wgpu/30.0.1/wgpu/struct.RenderPassTimestampWrites.html),
[timestamp period](https://docs.rs/wgpu/30.0.1/wgpu/struct.Queue.html#method.get_timestamp_period)
and [asynchronous mapping](https://docs.rs/wgpu/30.0.1/wgpu/struct.Buffer.html#method.map_async).

Allocator counters cover all Rust code using the executable's selected allocator,
including profiling overhead. They measure requested payload bytes; the peak is
a lifetime peak. Wasm memory is capacity, JS heap values are optional browser
estimates, and GPU payload counts exclude driver allocation details and framebuffer
storage. These scopes do not add up to total process memory. Allocator accounting
remains active while a profiling build is idle. Its forwarding wrapper is the sole
unsafe module: `engine_profile` uses `deny(unsafe_code)` with a local allocator
exception; the rest of the workspace retains `forbid(unsafe_code)`.

## Agent capture

Start a bridge against the profiling server:

```sh
cargo xtask agent serve --url http://127.0.0.1:5175/stress.html --port 7332
cargo xtask agent sessions --port 7332
```

Discover `{"op":"describe","name":"ProfileCapture"}` before capturing. With
the session paused, send this payload using the normal `agent request` command:

```json
{"op":"profile","duration_ms":5000,"max_events":65536,"cpu_samples":true}
```

The bridge clears input, resumes ordinary frames, captures, then pauses again.
It saves a trace JSON and optional `.cpuprofile` under
`artifacts/agent/SESSION_ID/`, returning paths and completeness. This advances
gameplay and never saves source; do not blindly retry an uncertain request.
`cpu_samples` defaults to `true`; disabling it avoids CDP sampling overhead. Its
sample interval is 1,000 microseconds and the `.cpuprofile` retains its own browser
profiler clock. See the official [CDP Profiler API](https://chromedevtools.github.io/devtools-protocol/tot/Profiler/).

Profiling Stress Lab exposes `window.__engineProfile.start(options)`, `status()`,
asynchronous `stop()` and `export()` for local automation. Start requires a running,
visible page and accepts the same duration/event bounds. The duration timer stops
automatically; manual stop preserves an incomplete capture. Export returns the
last report object after GPU settlement.

Observer coverage depends on capabilities reported in each artifact. The
[Long Tasks API](https://www.w3.org/TR/longtasks-1/) reports long main-thread tasks;
missing entries do not prove an absence of stalls. Compare the same seed, workload,
camera, resolution and artifact identity. Instrumentation, allocation accounting,
readback, observers, sampling and the HUD cost time; no fixed overhead percentage
is promised.
