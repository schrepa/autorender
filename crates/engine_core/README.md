# engine_core

Owner: engine maintainers. This crate owns the gameplay world, frame contract,
and CPU scene data. It depends on `engine_assets`, `engine_math`, `bevy_app`, `bevy_ecs`, and `glam`; it never
imports renderer, platform, browser, or GPU implementation types. The `math` module
(also in the prelude) holds the engine's branch-free trigonometry, including the
four-lane `sin_cos4`; see [engine math](../../docs/math.md).

`Engine::new(EngineConfig)` creates a Bevy `App` with `Camera`, `InputState`,
`ActionInput`, `FixedTime`, `FrameTime`, and `AssetLibrary` resources. The library initially
contains one cube mesh for the compatibility `Cube` component. Add ordinary Bevy `Plugin` implementations
and register systems against `FixedSimulation` and `Presentation`. The SDK
explicitly exposes Bevy types, so Bevy upgrades are public API upgrades. Plugins
must register before the first simulation frame; Bevy handles duplicate-plugin rejection.
Initial entities can be spawned in `Plugin::build` through `app.world_mut()`.
The engine also owns the `TransformPropagation` boundary. Bevy `Startup`, `Update`, `FixedUpdate`, runners,
and sub-app execution are not implicit extension points in this milestone.

## Frame contract

1. The host replaces `InputState` with one input batch and merges action bitsets with `ActionInput::ingest`.
2. `Engine::advance(delta_seconds)` runs plugin `finish`/`cleanup` once before its
   first frame. Plugins must report ready before that first call; asynchronous
   startup is not yet supported. `cleanup` is a startup hook, not scene disposal.
   Derived world transforms refresh before gameplay starts.
3. Zero or more `FixedSimulation` ticks run. `FixedTime` updates before each tick,
   and Bevy applies deferred commands at each schedule's end. A spawned entity is
   therefore visible to the next tick, presentation, and extraction. Dependencies
   within one tick require `before`, `after`, or `chain` ordering. Transform
   propagation follows each completed tick, so the next tick observes its results.
   Opted-in interpolation spans the pose before the tick and its final pose after
   propagation; changed sources update history during hierarchy validation.
   Optional `FixedPreSimulation` and `FixedPostSimulation` schedules surround
   gameplay. Nonempty boundaries apply commands and refresh globals before the
   next phase; post commands are included in final interpolation history. Empty
   boundaries add no transform scans. `engine_collision` uses these insertion
   points without introducing a dependency from core to collision.
   `ActionInput::just_pressed` and `just_released` expose edges in the first
   eligible fixed tick only. Edges survive frames with no tick; repeated edges
   coalesce. Held state survives all ticks. There are 32 typed action slots and
   no event queue allocations. Read edge flags during fixed simulation.
4. `Presentation` runs once, even on a zero-delta frame, and applies its deferred
   commands. `FrameTime` supplies frame deltas for camera movement and one-time
   pointer input. The host owns clearing or replacing frame input deltas.
   Transform propagation follows presentation, including frames with no fixed tick.
5. `Engine::extract(aspect, &mut snapshot)` culls mesh bounds and copies owned
   instance data. Each source used by emitted or culled instances is retained
   once through `Arc`; the caller then submits this to the
   renderer. It first refreshes derived transforms to include paused direct edits.
   No world borrow survives extraction.

Fixed simulation defaults to 60 Hz and at most five ticks per frame. Excess
whole ticks are discarded while the fractional remainder survives. A frame report
contains the performed ticks, dropped seconds, and interpolation fraction.
`extract` exposes exact current poses; `extract_with_presentation` can use the
fraction to interpolate opted-in local transforms. Invalid/negative input deltas become zero.
`FrameTime` separately caps its delta at 250 ms and accumulates those bounded
deltas. `reset_clock` clears the accumulator without rewinding simulation; the
host must also reset its timestamp after suspension. Reset also clears held and
pending action input, without synthesizing release events, and resets interpolation history.

The look's settings (`Atmosphere`, `Shadows`, `AmbientOcclusion`, `Bloom`,
`LocalLights`, `SunShafts`, `Reflections`, `Refraction`, `Particles`, `Decals`,
`ReflectionProbes`, `VirtualGeometry`, `LevelOfDetail`) are resources that also
serialize as their scene environment sections ([decision 022](../../docs/adr-022-scene-environments-and-agent-controls.md));
each checks itself with `validate` and clamps with `sanitized`.
`Environment::current` reads all of them, missing effect resources as off, and
extraction carries them sanitized as `RenderSnapshot::environment`. The optional
`schema` feature derives their JSON schemas for host tools.

The opt-in `dev` feature exposes `Engine::step_fixed(ticks)` for paused development
control. Counts outside 1–120 fail before mutation or plugin startup. A valid
request discards only the fractional accumulator, preserves pending action edges,
and runs exactly the requested fixed ticks regardless of the catch-up limit.
Deferred commands apply at each tick boundary and action edges appear only in the
first tick. Presentation runs once after the batch; the frame counter increments
once and presentation time advances by `min(ticks / fixed_hz, 0.25)` seconds.
Simulation time advances by the full requested duration, with zero reported
dropped time or interpolation. Use repeated one-tick requests if presentation is
needed between ticks. Reset the host timestamp before returning to real-time
advancement. `BrowserApp::dev_step` does this without clearing ingested actions;
`dev_render` extracts/submits current state without gameplay schedules or counter changes;
the engine-owned transform propagation boundary still runs.
Development stepping adds no protocol traffic or serialization to ordinary frames.

Configuration normalizes frequency into 1–1000 Hz (invalid values use 60 Hz),
and max ticks into 1–1000. Render extraction sets no instance budget by
default ([decision 029](../../docs/adr-029-budgets-and-device-limits.md));
`max_render_instances` sets one, and zero emits no instances while counters and
retention of culled sources still run.
Visible overflow is counted in `RenderSnapshot::dropped_instances`.
Extraction queries cache ECS metadata, copy only up to the instance budget, and
reuse snapshot vectors and private asset-deduplication sets. Ordering follows ECS iteration and is
not a persistent priority guarantee. Snapshot storage belongs to its caller;
CPU-to-GPU conversion and upload storage belong to the renderer.

The ordinary browser path uses `Engine::extract_frame(aspect, mode)`, returning a
borrowed `RenderFrame`. Its immutable `snapshot()` retains stable instance slots
and asset ownership; `visible_indices()` selects the submitted slots. Inactive
slots are not submitted. `model(slot)` reads each active slot's displayed matrix
from the prepared presented store through an index validated against the slot's
entity this frame; retained `RenderInstance::model` fields are not maintained.
`pose_rows()` holds every slot's pose in upload layout (`PoseRows`: the matrix's
transposed top three rows), rewritten exactly when the slot's pose stamp
advances, so a renderer uploads changed slots without re-reading matrices.
Frame identity, sequence, structure/pose/color revisions
and per-slot change stamps allow a renderer to reuse unchanged work. The borrow
prevents world mutation while consuming that frame. Source metadata resolves on
component/asset revision changes, and culling reuse checks the view and pose.
The renderer preserves those slot indices in GPU storage: 48-byte affine poses
and 16-byte colors have separate update streams, while grouped 4-byte slot IDs
select draws. Visibility, grouping and source metadata changes do not require
rewriting unchanged colors. Retained slot capacity includes inactive entries;
only `visible_indices().len()` is the submitted instance count.
The public owned-snapshot methods remain valid for independent snapshots that
must outlive later world edits. See [decision 013](../../docs/adr-013-canonical-poses-and-retained-rendering.md).

`Engine::extract_gpu_frame(aspect, alpha)` prepares an interpolated frame for
GPU presentation ([decision 014](../../docs/adr-014-gpu-presentation-and-culling.md)).
Its `gpu()` inputs (`GpuPresentation`) hold per-node current local poses
(`NodePose`, borrowed from the hierarchy's upload-layout cache), previous poses
where the consumer cannot copy them from its current ones (`shift`), parent links
and flags (`NodeInfo`), authority rows for nodes whose display does not depend
on the fraction, and active parents by depth, each record stamped with the gather
sequence that changed it and each stream's stamped count. Inputs are gathered only
after ticks, refreshes, resets or topology changes, not per displayed frame, and
an unchanged renderable structure skips the per-renderable pass. `visible_indices()`
lists every candidate, including invalid admitted nodes the GPU hides;
`slot_nodes()` maps slots to input nodes, and `culling()`/`instance_budget()`
describe the GPU's work. `model()` and `pose_rows()` are not prepared, and
counters list valid candidates as visible before GPU culling. The CPU path is
unchanged and remains the reference.

`Engine::run_ahead(predicted_delta)` runs one stage of the next fixed tick after a
frame is submitted, so at display rates of at least twice the tick rate each
frame runs at most one stage ([decision 015](../../docs/adr-015-pipelined-fixed-ticks.md)).
Until the tick is due, GPU frames present the previous interval;
`complete_pending_tick` completes a started tick (still hidden from GPU frames
until due, or until `reveal_ticks_ahead`), and exact extraction, stepping and
resets call it themselves. `tick_completed_ahead()` reports a completed tick that
is not yet due, and `gpu_current()` borrows the cached current poses with their
`GpuPresentation::current_version` and the stamp count of that version's gather
(`GpuCurrent`), so a renderer can stage them before the due frame gathers the
same version when that gather uploads them all. `BrowserApp` does all of this by default with GPU
presentation (`set_tick_pipelining`).

## Render interpolation

Attach the unit marker `InterpolateTransform` to local transforms animated by
`FixedSimulation`. Each interval spans the pose before the tick and its final
pose after deferred commands and transform propagation. Changed source values
update history during hierarchy validation. The first tick therefore starts
from the initial pose; an entity spawned during a tick starts from its current
pose. Multiple ticks retain only the latest interval; zero-tick frames preserve
that interval. Translation and same-sign scale interpolate linearly; normalized
quaternions follow the shortest arc. At absolute normalized dot products of at
least 0.9995, rotation uses normalized linear interpolation; larger rotations
retain spherical interpolation. This is an explicit small-angle approximation
for fractional display; see [decision 012](../../docs/adr-012-transform-validation-and-presentation.md).
Weighted translation avoids subtracting
opposite extreme endpoints. Conventional previous/current interpolation displays
one fixed tick behind wall time. Camera presentation remains immediate.

`Engine::extract_with_presentation(aspect, snapshot, PresentationMode::Interpolated
{ alpha })` prepares a fraction in `[0, 1]`. Nonfinite or out-of-range fractions
return `InterpolationError` before changing the engine or snapshot. Alpha zero
selects the previous pose; alpha one selects current. `PresentationMode::Current`
and the existing `Engine::extract` select exact current poses and preserve any
usable fixed history for later fractional previews.

Local interpolated matrices compose through the existing hierarchy, preserving
shear and reflection. Unmarked descendants inherit their displayed ancestors;
their own local values stay current. Both culling and draw records use this same
displayed matrix. Authoritative local and global poses are untouched. Read the
last prepared pose from `PresentedTransforms::get(entity)`: the result is absent
before preparation, and an admitted invalid pose has `matrix() == None`.
The store owns the canonical display matrix; no derived ECS output component is
required. Refreshing authoritative transforms alone preserves this display matrix.

Changes to tracked local poses outside a fixed tick snap that branch's local
histories, including presentation-system and paused direct writes. Marker/local
component removal and reinsertion, and entity generation changes, discard
incompatible history. Ordinary mutable queries, `edit_transform`, and replacing
an existing `Transform` with `insert` preserve its added tick.
Bevy's `World::modify_component` resets that lifetime; use mutable access or the
helpers for ordinary interpolated motion.
`set_parent`/`set_parents` reset tracked descendants of changed relationships,
including when the moved ancestor is unmarked. A tracked scale sign change snaps
the affected branch. Invalid authoritative nodes remain absent at every fraction;
interpolation cannot resurrect them. If a composed intermediate matrix fails the
normal-conditioning or finite-matrix contract, that node and descendants use
their authoritative current world matrices for that extraction.

Call `reset_interpolation(world, root)` after a teleport, including a teleport of
an unmarked ancestor. It resets tracked local histories on the root and current
descendants and returns their count. `reset_all_interpolation(world)` resets all
tracked histories. Both also suppress pending blends when called inside a fixed
tick; the next tick can capture a new interval normally. They do not advance
clocks or overwrite last-presented values. `Engine::reset_clock` resets all
history as part of suspension handling.

`Engine::presentation()` exposes `PresentationState`: last prepared mode,
`frame()`/`tick()` identity (absent before the first extraction), valid presented
node count, live history records, and memory observations. Frame/tick identify
CPU extraction, not GPU completion; advancing or inspecting without extracting
can leave them older than current gameplay clocks. After a GPU-presented
extraction, `gpu_pending()` is true and `PresentedTransforms` keeps its earlier
CPU values; `Engine::prepare_presented_transforms()` prepares that frame's exact
CPU reference on demand, and returns false without changes once a tick, refresh
or reset has changed its inputs.

History uses two reusable record buffers and a generational entity index, each
bounded by the transform-node limit. Each record is split into a cold identity part and a
hot 64-byte interval (previous endpoint, epoch and validity) that ordinary motion
and presentation touch alone. Hierarchy validation forwards changed source poses,
component lifetimes and validity through cached history slots. A record freezes
its previous endpoint once per fixed epoch, while the hierarchy's local-pose
array owns the current endpoint. Presentation reads that current pose without keeping
a second copy in each history. An unchanged tick expires the preceding motion interval
without scanning every record. Normal tick begin/end bookkeeping is O(1);
validity changes use a bounded coalesced list, and epoch rollover performs a rare
bounded reset. Structural reconciliation checks membership/order in one query
pass, retaining matching records and the already updated prefix when rebuilding
the index. These storage changes preserve the same interpolation boundaries;
see [decision 011](../../docs/adr-011-in-place-transforms.md).
A node's history hint and its record's hierarchy index form a pair rewritten at
every topology rebuild; hot paths accept a hint only when both agree and otherwise
resolve the full generational entity through the membership index. Sparse composition traverses blended roots and descendants
using existing child links. At 50% or more opt-in histories relative to cached
hierarchy nodes, preparation walks the existing parent-first order and blends
directly during composition using validated source-to-history hints. There is no
separate dense history scatter pass or touched-index list. Four eligible local
pose pairs at a time use component-major SIMD blending in bounded stack arrays;
parent-first composition still applies inherited reset and fallback decisions
sequentially. Larger rotations and threshold neighbors use the scalar rotation
policy. These numeric blocks add neither heap allocations nor persistent
structure-of-arrays pose storage. This avoids intermediate full-scene local-pose storage and
repeated ancestor/work-stack visits for broad motion. The threshold is a
provisional workload policy, not a measured universal crossover.

Both paths share the blend policy and error budget, reset propagation and matrix
validity/fallback checks; packed and scalar results need not be bit-identical on
every backend. Dense preparation initializes flags in its parent-first pass; sparse
preparation resets previously touched flags. A dense-to-sparse transition clears
the former dense flags. Private matrix payloads use separate validity flags and
are assigned before becoming active. Sparse preparation retains its compact
blended-local vector. Composition publishes directly into the canonical presented
store; no ECS output-copy or component-repair pass follows it. Validated cached
indices check full entity identity before reuse. No additional topology graph is retained.
New peak populations grow retained capacity; removed records leave live storage
while capacity remains reusable. Without markers, preparation records authoritative
matrices without allocating history or blend traversal buffers. Exact-current mode
skips interpolation composition. Fixed capture bookkeeping normally costs O(1),
with changed-source work shared with hierarchy validation. Fractional
preparation uses bounded ancestor checks and affected-branch composition for
sparse motion, followed by a contiguous O(admitted nodes) scan to retain a
complete last-prepared snapshot. Dense motion uses an O(hierarchy nodes) ordered
pass. Neither path publishes derived ECS components. Measure matched browser workloads before
making performance claims.

`history_capacity()` sums both record buffers; `history_bytes()` measures those
Vec payload capacities, including hot intervals and node hints. `presentation_scratch_bytes()` measures temporary
blended-local and traversal Vec capacities; `presented_store_bytes()` measures
canonical matrix/revision/flag/entity Vec capacities, including invalid entries.
Hierarchy and presentation storage keep one array per field group in a shared
node index space (local poses, traversal links, matrices, revisions, flags), so
each pass streams only the fields it reads.
These observations exclude entity HashMaps, query metadata, allocator overhead,
and unused Bevy archetype capacity; they are not total process-memory estimates.

## Mesh and material contract

Insert immutable `Mesh`, `Texture`, and `Material` values into `AssetLibrary` and attach their
typed handles with `Mesh3d` and `Transform`. A material supplies opaque linear
color and an optional base color texture handle; a transform supplies position, normalized rotation and scale. CPU asset
validation and storage limits belong to `engine_assets`. Handles are weak,
generational, library-specific runtime references; they are not save identifiers.
The legacy `Cube` component uses the engine's built-in mesh and its inline color.
`Mesh3d` takes precedence when both components exist on an entity.

Extraction counts absent meshes/materials/textures in `RenderSnapshot::missing_assets`
and omits those entities. It counts one missing entity even if both handles are
invalid. Missing entities consume no instance budget. The `meshes` and `textures` vectors contain
only unique sources used by emitted or frustum-culled valid instances; unused
library assets and sources used only by dropped instances stay out of the
snapshot. Extraction clones one `Arc` per unique mesh/texture and copies each
emitted instance's matrix/color/mesh/texture handle. Snapshot source metadata is
bounded by the originating library's admitted asset count/payload budget,
independently of the visible instance limit;
the renderer independently validates GPU residency budgets. Removal invalidates future lookups while
previous snapshots remain usable; release old snapshots to retire those retained
CPU sources. Every renderable is visited to account for invalid transforms,
missing assets, culling, and visible overflow.

## Frustum culling

`EngineConfig::frustum_culling` defaults to true. `Engine::set_frustum_culling(bool)`
and `frustum_culling()` expose a temporary runtime toggle, including an unculled
baseline for repeatable comparisons. The toggle changes extraction only; it does
not change simulation, clocks, authored source, or asset ownership.

Culling derives six planes from the same camera view-projection matrix sent to
the renderer, using WebGPU's `-w <= x,y <= w` and `0 <= z <= w` clip ranges.
Each valid mesh uses its immutable local `Aabb`. Transforming a plane into local
space allows an exact box support test for rotations, nonuniform scales, shear,
and negative scale without decomposing the global matrix or scanning vertices.
An intersecting or enclosing box remains eligible. Culling has no occlusion test.
A world-space interior test first accepts boxes proven strictly inside all six
planes; only intersecting or unproven boxes need the transformed-plane test.
Decisions are identical to the transformed-plane test alone.

Extraction retains visibility results for its valid input rows.
Exact model bits and immutable mesh handles reuse a result only while the camera
matrix and culling mode match. Query reordering or entity replacement is safe:
reuse depends on equal geometric inputs, not entity identity. Asset lookups still
run before reuse, so removed dependencies cannot remain visible from the cache.
Output snapshots are rebuilt as owned data, including when a caller changes or
replaces its previous snapshot. The cache holds weak handles and matrix bits,
adds one entry record per input row, and retains capacity after removals. Profiling
reports its capacity and the number of culling tests performed/reused.

The support test uses a conservative margin of 32 f32 epsilons times a bound on
absolute model/projection products, with an absolute floor. Geometry or matrix
coefficients above `1e15`, nonfinite intermediates, and product bounds approaching
overflow fail open: those instances remain visible. This bounds subnormal-flush
amplification within the fast arithmetic path and avoids ambiguous rejection for
extreme finite assets. The threshold is a culling shortcut, not an asset or world
transform admission limit. No normalization, per-entity allocation, string lookup,
or vertex traversal enters culling. Performance must be measured in browser workloads.

Snapshot counters partition renderable candidates (entities with `Cube` or
`Mesh3d`) in this order: invalid transform, missing asset, culled, then visible.
`render_candidates = invalid_transforms + missing_assets + culled_instances +
visible_instances`. Visibility precedes the output budget, so
`instances.len() + dropped_instances = visible_instances` for owned snapshots.
For borrowed retained frames, use `visible_indices().len()` in place of
`instances.len()`, because stable storage includes inactive and culled slots.
Culled entities consume no submitted-instance budget. Disabling culling makes
every valid candidate visible.

The snapshot retains mesh/texture sources for both culled and emitted instances.
The renderer uses culled sources only to retain resources already resident;
offscreen-only assets do not trigger first uploads. Returning to view can reuse
existing GPU assets, while actual source/entity removal still permits retirement.
Invalid, missing-asset and budget-dropped instances alone retain no sources.

## Transform hierarchy

`Transform` contains local translation, normalized rotation, and scale. Roots use
the local matrix; children use `parent_global * local`. Read a full world matrix
through `Res<GlobalTransforms>::get(entity)` or
`world.resource::<GlobalTransforms>().get(entity)`. It preserves shear caused by
rotations below nonuniform scale. A missing store entry returns `None`; an admitted
invalid entry returns a `GlobalTransform` whose `matrix()` is `None`.

`Transform` remains an ordinary mutable Bevy component. Use `Query<&mut Transform>`
for frequent gameplay updates; direct World mutable access is also tracked.
`GlobalTransform` and `PresentedTransform` are copyable read values, not components.
Their engine-owned resources expose no matrix mutation API. `Parent` stays
immutable and uses the checked helpers below. No special plugin registration or
app wrapper is required. See the [typed-read migration](../../docs/migrations/0.1-canonical-poses.md).

Systems edit poses through ordinary `Query<&mut Transform>`.
`edit_transform(world, entity, closure)` immediately edits an existing
pose in place, returns `MissingEntity`/`MissingTransform` when appropriate, and
skips a bit-identical result. It does not emit insertion observers. Refresh
globals before inspecting derived data after an immediate edit. Normal
`World`/`Commands` insertion, replacement and removal retain Bevy's lifecycle
behavior. See [decision 011](../../docs/adr-011-in-place-transforms.md) for migration.

Use `set_parent(world, child, Some(parent))` or `set_parent(world, child, None)`.
These preserve the local values; detaching does not preserve the previous world
pose. `Parent::parent()` reads the relationship. `set_parents(world, &changes)`
validates and installs a complete batch atomically, suitable for scene loading.
Both reject absent entities/transforms, cycles, and depth over 64 parent edges.
Moving a subtree validates its descendants too. Batch edits allocate bounded
temporary topology storage and perform O(N + changes) work outside the frame loop.
Repeated child IDs fail. Invalid unrelated branches do not block a repair.

Parents do not own their children. Despawning a parent leaves its children alive
and invalid until explicitly detached or reattached. This keeps scene disposal
under its explicit owner's control. Entity generations prevent references from
binding to another entity that later reuses the same index.

The named `TransformPropagation` schedule runs before frame gameplay, after each
fixed tick, after presentation, and before extraction. Within gameplay schedules,
globals reflect the preceding boundary; use local data and explicit system ordering
for same-phase edits. `Engine::refresh_transforms()` runs only this boundary;
`propagate_transforms(&mut World)` supports immediate refresh during transactional
scene installation and returns counts. Neither advances clocks or gameplay.

Propagation admits `TransformNodeLimit` transformed entities (by default every
one a 32-bit index can address; `EngineConfig::max_transform_nodes` sets a
budget). Ordinary boundaries apply an
engine-owned change window to each table's contiguous `Transform` change stamps:
a table whose stamps are all unchanged is skipped in one pass, and others classify
blocks of stamps before fetching any payload. Unchanged rows load no payload.
Changed rows match a dense copy of source entities and read their cached source
record only when the `Transform`'s added tick also lies in the window (a re-added
`Transform`, such as after `World::modify_component`). The decision equals
Bevy's `Changed<Transform>` test for the same window, which is independent of
`World::clear_trackers`.
A completed refresh acknowledges its source scan, then advances the ECS
change tick so successive direct writes remain visible without advancing frame
or simulation clocks. First use, structural changes, interrupted propagation,
aged timestamps and populations at/above the admission limit force authoritative
source validation. Tick-maintenance notifications also invalidate the cache.

Local poses retain exact bit comparisons; canonical global/presented values expose
output revisions that change with matrix or validity changes and may change
conservatively during topology rebuilds or reindexing. A changed leaf whose
parent's admission certificate proves it valid defers its world matrix
([decision 016](../../docs/adr-016-deferred-leaf-globals.md)): reads compose it
from the parent's stored matrix and the cached local, bit for bit as eager
evaluation, and its revision changes whenever it is re-evaluated, which can be
conservative when the composed bits did not change. `u64::MAX` is an
uncacheable exhaustion sentinel. `TransformIndex` hints let retained consumers
use `get_indexed_with_revision` without assuming a stable slot. Compact source
records remain separate from derived matrices and traversal metadata. Stable
topology retains indices and child links across ordinary motion. Changed rows
resolve source indices by full Entity identity, so unrelated archetype moves do
not require rebuilding topology. Structural changes reconcile membership/order,
while invalid local-pose repair remains independent of ancestry validity.

Global propagation publishes changed canonical values without an ECS output pass.
Use the named boundaries or store output revisions to observe world motion;
`Changed<GlobalTransform>` no longer applies. Explicit local replacements (`insert`)
still invoke ordinary lifecycle observers. Pending observer commands are flushed
before source scanning. No schedule or plugin extension boundary is skipped.

`bypass_change_detection` and manually rewriting change ticks deliberately opt
out of Bevy tracking and are unsupported for transform updates. Ordinary mutable
queries and direct World writes require no manual dirty marking. Profiling
distinguishes metadata scans, payload validation and matrix evaluation;
browser captures establish any timing benefit.

[Decision 012](../../docs/adr-012-transform-validation-and-presentation.md)
documents the sufficient matrix-admission bound, dense presentation pass fusion,
and compact private matrix storage. The fallback retains the original numerical
predicate. Public matrix accessors continue returning `Mat4` values.

Invalid local values, nonfinite/singular composed matrices, invalid normal
cofactors/inverses, orphaned parents, cycles, depth overflow, and capacity overflow
invalidate affected global matrices; healthy unrelated branches still update.
`valid_global_matrix` is shared with authored validation. Extraction skips invalid
renderables and counts them in `RenderSnapshot::invalid_transforms` before testing
assets or charging instance capacity. `HierarchyReport` separately counts all
transformed entities, invalid nodes, and capacity overflow.

Normal transformation also uses a conservative conditioning limit: after scaling
the cofactor matrix's largest absolute coefficient to one, its inverse infinity
norm must be at most 10,000 (`MAX_NORMAL_INVERSE_NORM`). This rejects severe
anisotropy or nearly singular shear, including some previously accepted extreme
version 1/2 transforms. Finite inversion alone does not establish safe GPU normal
arithmetic. The shader rescales basis/cofactor data and the resulting normal
before normalization; large or small uniform scales that pass the existing
finite-matrix checks remain supported. Native arithmetic checks establish this
contract, not a guarantee about every browser's floating-point rounding.

The browser baseline uses one execution context and a single-thread executor.
Scheduling data access and explicit dependency edges remain Bevy's contract.
World ownership is exclusive; don't replace `Engine::app`'s world or remove the
engine resources/schedules. Camera coordinates are right-handed, Y-up, with a
WebGPU 0–1 depth projection.

Native tests cover catch-up bounds and remainder retention, invalid clocks,
deferred visibility, presentation timing, bounded reusable extraction, mixed
geometry/material resolution, and snapshot lifetime after asset removal. Hierarchy
tests cover composed shear, atomic relationship edits, subtree depth, orphan and
generation handling, invalid graph isolation, fixed/presentation/extraction
boundaries, exact development ticks, cache reuse, and capacity recovery.
Culling tests cover all planes, WebGPU depth, grazing/intersecting bounds,
hierarchy edits with shear/reflection, overflow fail-open, budget/counter
partitions, retained source ownership, and 5,000 deterministic transforms checked
against an independent double-precision clip-corner reference.
Interpolation tests cover first/zero/multiple ticks, shortest-arc rotation,
hierarchical shear, explicit and mid-tick resets, reparenting, marker/component
replacement, generational identity, invalid intermediate fallback, displayed
culling, extraction identity, invalid-alpha atomicity, dense/sparse agreement,
cached-index invalidation, and bounded storage reuse.

`FixedPreSimulation` and `FixedPostSimulation` surround each `FixedSimulation`
tick. Their deferred commands and authoritative transform propagation finish
before the next boundary; post systems finish before interpolation captures its
current endpoint. These are generic extension schedules. Empty pre/post phases
avoid extra transform passes; core imports no collision implementation. See
[decision 009](../../docs/adr-009-collision-and-spatial-queries.md).
