# Decision 014: GPU presentation and culling

Status: adopted. Ordinary interpolated browser frames blend, compose, validate
and cull displayed poses on the GPU. The CPU presentation pass of decisions 008,
012 and 013 remains the exact reference for captures, previews, inspection,
public snapshots and every non-browser consumer. Authoritative simulation,
collision and gameplay reads are unchanged.

## Context

With a 60 Hz fixed tick, most displayed frames on a 120 or 240 Hz display show a
new fraction of the same interval. The CPU path nevertheless blended, composed,
validated and culled every interpolated node, rewrote pose rows and uploaded them
on every frame. Round-3 browser profiles of the 90,000-object all-moving Stress
Lab workload spent 6–7 ms per frame outside fixed ticks on that work, repeated
between ticks. It is data-parallel per node and depends only on the previous and
current local poses, the fraction and the parent's displayed pose.

## Inputs and frames

`Engine::extract_gpu_frame(aspect, alpha)` refreshes transforms, then gathers
per-node inputs in the hierarchy's parent-first order, mirroring the dense CPU
pass's decisions:

- Current local poses (`NodePose`, 40 bytes: translation, scale x, rotation,
  scale y and z). The hierarchy caches every admitted local in this layout, and
  the inputs borrow that cache instead of copying it. A record is stamped for
  *active* nodes, whose display the GPU composes from locals (those that blend,
  those below an active parent, and valid nodes whose local changed since the
  last gather, composed from their current local without blending: uploading
  that 40-byte local is cheaper than composing and uploading authority rows),
  when its local changed since the last gather or the node just became active. Because the poses are borrowed, a consumer must
  apply each gather before a later boundary changes the hierarchy: frames that
  present an in-flight tick upload nothing. `BrowserApp` tracks whether the
  renderer submitted (`RenderStats::submitted`); after a frame it skipped
  (suspended, occluded or recovering surface) the next GPU frame completes and
  presents the latest tick so its catch-up upload is consistent.
- Previous local poses (`NodePose`). After exactly one fixed tick with no other
  local change since the last gather, every interval starts at the current pose
  that gather published, so the gather sets `shift` and consumers copy their
  current poses into the previous slots on the GPU; only current poses upload.
  Other gathers stamp explicit previous poses of blending nodes. A consumer that
  missed the copy stops blending until a later shift it applies.
- `NodeInfo` (8 bytes): parent index and flags (`NODE_VALID`, `NODE_BLEND`,
  `NODE_ACTIVE`, `NODE_PARENT_ACTIVE`). Snap and reset decisions propagate to
  descendants exactly as on the CPU; invalid ancestry clears every flag.
- Authority rows (48-byte `PoseRows`) for valid inactive nodes (static since the
  last gather, not blending, not below an active parent), written only when that
  node's authoritative output revision changed.
- Active nodes with children, sorted by depth, with one end offset per level.

Each record carries the gather sequence that last changed it, and the gather
publishes how many records it stamped per stream (`GpuPresentation::stamped`)
and, while they number at most a quarter of the nodes, which ones
(`GpuPresentation::stamped_nodes`).
A gather runs only when the topology revision, hierarchy output revision or
interpolation revision differs from the last gather, normally once per fixed
tick (the interpolation revision advances at every tick boundary). A topology rebuild
starts a new input identity, which consumers treat as a full upload. Unchanged
local pairs keep their stamps, so a static child of a moving parent is not
re-sent.

A gather visits only the nodes whose inputs can differ from the last gather,
so its cost follows the changes rather than the node count. Propagation lists
every node whose cached local or output revision changed since the last gather
(`GlobalTransforms::changes`, at most a quarter of the nodes). The gather then
visits, depth by depth, the nodes active at the last gather, every
interpolation record's node (blend decisions follow the tick), the listed
nodes, and the children of every visited node whose snap or active decision
changed. Any other node was inactive, has no record and an unchanged parent,
so it stays inactive with unchanged authority rows. A new node space, an
overflowed list, or more than a third of the nodes to visit (counted once
each: active, listed and recorded nodes largely coincide) walks every node
in parent-first order instead; so do the two gathers after a rebuild (the first
activates every node, the second settles them). Test builds check every sparse
gather against a full walk of the same state (`interpolation.gpu_visited`
counts visited nodes). A frame that only changes the fraction performs no gather and, while
renderables, sources, assets and topology are unchanged, skips the retained
per-renderable pass (`extraction.gpu_structure_reused`); only the camera is
updated. GPU frames list every candidate in `visible_indices()` and expose
`gpu()`, `slot_nodes()`, `culling()` and `instance_budget()`. They do not
compute culling, `pose_rows()` or `model()`; those are unspecified for GPU
frames.

## GPU passes

The renderer uploads records whose stamp is newer than the sequence it last
uploaded, as at most 64 coalesced ranges (a full write beyond 75% density),
through the mapped staging ring for 256 KiB or more and `writeBuffer` otherwise.
One gather behind, the published counts decide instead: a stream without stamped
records is skipped and one at least 75% stamped is written whole, without
scanning its stamps; other streams plan the same ranges from their listed
nodes through a node bitmap when the gather listed them. A gather that sets
`shift` is preceded by a separate `shift_previous` dispatch submitted before its
uploads. It copies the current pose of every node the last applied gather made
active (its flags are still in the info buffer): only those published a pose,
and a gather sends any other blending node's previous pose explicitly.

Current poses can also be staged a frame early. `Engine::gpu_current` exposes
the hierarchy's cached locals with `GpuPresentation::current_version` (topology
and hierarchy revision); equal versions certify equal poses. When a pipelined
tick completes ahead of its due frame ([decision 015](adr-015-pipelined-fixed-ticks.md)),
`BrowserApp` passes them to `Renderer::prestage_gpu_current`, which copies them
into a mapped staging buffer right away. The due frame's gather then has the same
version, and when it uploads every current pose the renderer records a GPU copy
from that buffer instead of staging them again (`render.gpu_current_prestaged`).
Any other outcome drops the staged buffer back to the ring unused, so the renderer
stages ahead only when the gather of that version stamped at least three quarters
of the current poses (`GpuCurrent::stamped`): a mostly static scene uploads its few
changed poses in the due frame instead of copying every pose. This moves the
largest per-tick CPU copy (3.6 MB at 90,000 moving objects) from the frame that
presents a tick to the frame that completes it.
Candidate and group records (slot, node, draw group and position in draw-group
order; group start, count, index count and local bounds) upload only when the
retained structure, source revision or input identity changes. Candidates upload
twice: in draw-group order for compaction, and in slot order for culling. Each frame then writes one 192-byte uniform (fraction,
counts, budget, culling flag, frustum planes and the absolute view-projection)
and records one compute pass (profiling captures time each stage in its own
pass; see `docs/profiling.md`):

1. `compose_parents`, one dispatch per depth level, composes active parents in
   parent-first order into a per-node buffer of displayed and current matrices.
2. `cull`, one thread per candidate in slot order, composes its node over that
   parent (or over authority rows), validates it, writes the slot's pose rows
   into the retained pose buffer, tests visibility, picks its level of detail
   (decision 047), and writes the flags at the candidate's draw-group
   position. Slot order follows node order closely, so
   pose, node and authority records stream; in draw-group order each group's
   candidates sit far apart. At 250,000 objects in eight meshes and textures
   this halved the stage (2.19-2.41 ms to 1.18-1.23 ms, Firefox, Intel Xe-LPG),
   about the cost of a single draw group.
   `sum_groups` then reduces the flags of every 64 candidates in draw-group
   order, counting each level of detail in its own lane.
3. `scan_groups` turns group sums into exclusive offsets and totals.
4. `batch_args` writes one indirect draw per draw group and level, applying the
   instance budget, and each group's first visible offset per level.
5. `scatter` compacts visible slot IDs into each group's region of its level's
   plane in candidate order.

Each group draws with `draw_indexed_indirect`, reading its region of the visible
ID buffer as the instance vertex stream. Devices with `indirect-first-instance`
bind the buffer once and the arguments name the region as the first instance;
others bind each group's region (decision 046). The render pipeline, pose storage and color storage are shared with
CPU frames. A four-byte total is copied to one of three readback buffers and
mapped asynchronously; nothing waits on the GPU.

## Numerical contract

The shader mirrors the CPU policy in f32: glam's scale-rotation-translation
matrix construction, normalized linear rotation blending at an absolute dot of
at least 0.9995 and spherical interpolation below it, exact endpoints at fractions
zero and one, and parent-first composition. Validation uses the common sufficient
admission bound of decision 012 (largest axis coefficient in [2^-10, 2^20],
translation magnitude at most 2^20, determinant at least 2^-9 times the cubed
scale). A blended matrix outside it displays the node's current composed pose,
and that fallback is inherited by its active subtree. The CPU validator's exact
path can admit such matrices; this is a display-only difference confined to
extreme scales or coordinates beyond about one million units. Invalid
authoritative nodes are never drawn.

Culling uses the same conservative transformed-plane support test, rounding
margin, coefficient limit and fail-open product bounds as core; shared constants
are written as exact hex literals and tested bit-for-bit against the CPU values.
Results agree with the CPU reference within f32 rounding: WGSL and glam may round
individual operations differently. `gpu-presentation-smoke.mjs` compares CPU and
GPU captures of the same fractions and requires equal visible counts within 0.1%.

## Observable behavior

`BrowserApp` presents ordinary `Interpolated` frames on the GPU unless
`set_gpu_presentation(false)` is called. Current-mode frames, exact `step`,
`render` and `capture`, `present {alpha}` previews and benchmark `present` mode
use the CPU path; development `present`/`capture` accept `gpu: true` with an
alpha to preview the GPU path instead.

`PresentedTransforms` keeps its earlier CPU-prepared values after a GPU frame.
`PresentationState::gpu_pending()` reports this, and
`Engine::prepare_presented_transforms()` prepares that frame's exact CPU reference
while its inputs are unchanged; a later tick, refresh or reset makes it return
false without changes. Development inspection materializes a pending frame before
refreshing transforms, so `PresentedTransform` remains the last extraction's pose.
Observations include `gpu_pending`.

GPU-frame counters still partition candidates into invalid, missing-asset,
culled and visible. Invalid admitted nodes stay listed, because validity can
change without a structural rebuild, and the GPU hides them. `visible_instances`
is the pre-budget count read back from the latest completed frame, a few frames
old (`RenderStats::gpu_visible`); before the first read-back every valid listed
candidate counts as visible. Culled counts are the remainder, and `instances` is
the visible count limited by `max_render_instances`. The GPU applies that budget
in draw-group order, so the submitted count matches the CPU path but a different
subset can be dropped.

## Resources and limits

Per-node buffers hold 40-byte current and previous locals, 8-byte info and
48-byte authority rows,
plus 112 bytes per node for composed parents only when active parents exist.
Candidates use 16 bytes each, plus scan words and a visible-ID word per level
of detail; groups use 96-byte records (with their mesh's place in the mesh
arena, decision 046, and levels, decision 047), base offsets and 20-byte
indirect arguments per level. Buffers grow to powers of
two (at least 1,024 nodes or candidates and 64 draw groups) and never shrink;
`memory.gpu_presentation_bytes` reports them and `memory.gpu_input_bytes` the
CPU-side inputs. Depth levels use one 16-byte
uniform each (at most 65) with their own bind groups, because automatic layouts
have no dynamic offsets. The pass requires eight storage buffers per shader
stage, the WebGPU default. Scanning handles any group count in 512-group chunks
of one workgroup.

Pipeline creation is asynchronous. In a fresh Chrome GPU process, compiling the
compute pipelines took about 0.5 s (`compose_parents` about 0.2 s, `cull` about
0.25 s); a second device reused the cache in under 50 ms.

## Measurements

Profiling build, headless Chrome on a Core Ultra 9 275HX pinned to performance
cores, the user's 90,000-object all-moving configuration and camera, one build
with `?presentation=cpu` as the reference. Frame rates were measured without vsync
or frame-rate limits using the engine's own frame and tick counters. Runs
alternated in ABBA order with paired ratios. See
`artifacts/gpu-presentation-2026-09-25.md` for the reports.

| Adapter | GPU / CPU presentation fps | Presentation compute pass (GPU) |
| --- | --- | --- |
| NVIDIA RTX PRO 5000 (forced) | 1,284 / 82 (paired ×16.1) | 0.06 ms |
| Intel Xe-LPG (Chrome default) | 276 / 87 (paired ×3.15) | 1.0 ms |

On the NVIDIA adapter, frames without a fixed tick cost 0.1–0.2 ms of CPU at
every measured size; on the CPU path they cost 4.3 ms at 60,000 objects and
6.2–7.0 ms at 90,000. Frames containing a tick still run simulation, propagation
and collision, plus the input gather and upload: at 90,000 all-moving objects
about 2 ms of gather and 1.3 ms of upload per tick. Tick-frame medians were
3.0–3.3 ms at 20,000 objects, 4.2–4.7 ms at 30,000, 7.2–8.2 ms at 60,000 and
10.6–11.6 ms at 90,000, against 9.5–9.7 ms and 13.8–17.4 ms on the CPU path at
60,000 and 90,000.

Ticks themselves measured 10–20% slower with GPU presentation in most paired
runs, including the unchanged gameplay system, consistent with this laptop's CPU
yielding clock and power to a busier GPU (and with frames no longer warming
simulation data in cache). The per-frame saving exceeds that cost. Vsync-paced
60 Hz captures on the integrated GPU cut CPU time outside ticks from 6.4 to
3.4 ms per frame (paired −44%): at one tick per frame each frame also gathers
and uploads. Treat these as local headless measurements, not a portable
frame-rate promise.

### Firefox at 120 Hz

Later work reduced the per-tick CPU cost of this path in Firefox 156 (same
configuration, headful, 120 Hz vsync; ABBA sessions of three page loads, see
[decision 015](adr-015-pipelined-fixed-ticks.md) and
`artifacts/firefox-120hz-2026-09-25.md`). Main-thread work per fixed tick fell
from 15.7–16.0 ms to 13.5–13.9 ms; per tick:

- Lean inputs (current and previous poses of 40 bytes each, the GPU copy of
  previous poses): gather 4.5 to 2.3 ms and upload 7.4 to 3.7 MB.
- Evaluating changed leaves during the change scan (decision 012): native
  propagation 4.5% faster. A lane-parallel but bit-identical `Transform::matrix`
  and a vector admission gate: propagation 0.2–0.35 ms less (in-page A/B).
- Published stamp counts: renderer CPU 1.5 to 0.85 ms (no stamp scans).
- A 16-byte hot interval record (previous poses moved to the cold record, with a
  cached scale-compatibility bit): gather 2.3 to 1.8 ms.
- Current poses borrowed from the hierarchy's upload-layout cache: gather 1.8 to
  1.2 ms, with propagation no slower.
- Collision BVH refit for unchanged membership: post-simulation 0.6 to 0.3 ms.

Remaining work per tick: gameplay about 5 ms (the workload's own trigonometry),
propagation 5.5–5.8 ms, gather 1.2 ms and renderer 1.3 ms across two frames.
Attempts that did not help in Firefox were reverted: evaluating leaves four per
SIMD lane group (5–10% slower per tick in-page, in both browsers), writing GPU
poses during propagation (moved 0.45 ms rather than saving it) and splitting the
gather across frames (decision 015).

## Consequences

Per-frame cost no longer scales with the number of interpolated objects.
Smoothness at high refresh rates is now bounded by the frame that runs a fixed
tick. Average frame rate rose three- to sixteen-fold, but the object counts at
which every measured frame fits a 240 or 120 Hz budget rose only 10–25% (to about
25–28k and 55–58k moving objects on the NVIDIA adapter). Reducing tick cost,
uploading only changed current poses, or moving simulation off the presentation
thread are separate decisions requiring their own measurements.

Future render features must accept GPU-written pose storage in slot order and
treat the candidate list, not a CPU visible list, as the input to drawing. Keep
the CPU path as the reference and opt-out. Display poses never feed back into
simulation, collision or authored data; gameplay that needs presented poses uses
`prepare_presented_transforms` or CPU presentation explicitly. The GPU path adds
no worker threads, asset formats or render graph.

## Alternatives and evidence

- **Hierarchical culling: blocks first** (considered 30 September 2026).
  Testing blocks of candidates before their members only pays where the
  per-candidate tests cost something. They do not, measured in the Stress
  Lab Metro overview (Firefox, integrated GPU, 90,000 candidates).
  - `presentation_cull` takes 0.64 ms.
  - Marking every candidate visible at level 0, with no frustum, sub-pixel
    or level tests, takes 0.65 ms.
  - The pass is bound by each candidate's reads and its pose write, which
    shadow casters need for every candidate whether the camera sees it or
    not. Blocks could not skip that work.
- **A persistent static scene** (sized 1 October 2026). Still instances
  repeat the same work every frame: culling rewrites poses that did not
  change, and shadow selection hashes them to find that they did not move.
  Shader-only variants sized that work in the same overview (three
  alternating Firefox runs each; 86,700 of the 90,000 candidates are in the
  frustum):
  - Not writing inactive nodes' poses: `presentation_cull` 0.654 to
    0.564 ms. Without temporal antialiasing culling writes the retained
    instance buffer, which already held them, so images did not change.
  - Writing them only where the stored rows differ: 0.625 to 0.598 ms (a
    second session). Reading costs about what writing does.
  - Inactive candidates skipping culling altogether, which could only hold
    while the camera stands still: 0.40 ms.
  - Shadow selection returning after the stillness read for kept casters:
    `shadow_select` 0.49 to 0.38 ms. The rest is reading every candidate's
    records.
  - Flags in slot order instead of at draw-group positions (second
    session): 0.016 ms in culling and 0.004 ms in shadow selection. The 360 KB flag array stays
    in the GPU's cache, so scattered flag accesses cost nothing.
  - Counting still frames for probes: about 0.02 ms. The occlusion test is
    0.20 of the occlusion pass's 0.77 ms; the depth pyramid and phase two's
    compaction are the rest.

  With a moving camera every test stays, so the attainable total is about
  0.2 to 0.35 ms of a 21 ms frame, and blocks would skip little more where
  nearly every candidate is in view. Not built.
