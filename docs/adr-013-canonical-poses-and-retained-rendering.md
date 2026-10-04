# Decision 013: canonical poses and retained rendering

Status: adopted. Core owns canonical authoritative and presented poses. This
supersedes the derived-component storage and observer contracts in decisions
006, 008 and 011, while retaining decision 012's numerical policy. Public gameplay
continues to use Bevy entities, local `Transform` components and named schedules.

## Canonical pose ownership

The engine already derives world matrices into its hierarchy cache. Copying every
changed matrix into a second ECS component column adds writes, change tracking
and repair work without creating new gameplay information. Presentation had a
similar second output column and validation pass. Canonical stores make the
engine-owned result the readable result.

`GlobalTransforms` and `PresentedTransforms` are resources with private storage.
Their `get(Entity)` methods return copyable `GlobalTransform` and
`PresentedTransform` values. These values are not components and cannot be used
to mutate the stores. A missing result means no admitted/prepared entry; an
invalid returned value has no matrix. Full affine matrices retain shear and the
existing finite, singularity and conditioning rules.

Frequent local motion still uses ordinary mutable Bevy queries. Transform
membership, parent changes, entity generations and change ticks continue to
control propagation and interpolation history. Callers do not manually mark
poses dirty. Derived-component insertion/removal hooks and repair paths are no
longer part of the contract. `Parent` remains immutable and checked parent helpers
retain their topology validation.

The stores provide a validated `TransformIndex` hint for retained consumers.
`get_indexed_with_revision` also returns an output revision, which changes when
the stored matrix or validity changes; topology rebuilds or reindexing may also
conservatively change it, as may re-evaluating a leaf whose matrix is deferred
([decision 016](adr-016-deferred-leaf-globals.md)). A revision of `u64::MAX` is uncacheable after stamp
exhaustion. The hint is an optimization, not ownership or a stable authored
identifier. Invalidated topology or a reused entity slot cannot make a retained
consumer read another entity's pose.

Ordinary boundaries read each table's contiguous `Transform` values and change
stamps through Bevy's `contiguous_iter`. With `since = min(this_run - last_run,
MAX_CHANGE_AGE)` (wrapping), a stamp `t` changed exactly when
`this_run - t < since`, which is Bevy's `Tick::is_newer_than` for that window.
Blocks of 16 stamps are classified branch-free first, so unchanged boundaries pay
one pass over stamps and no per-row query fetch. Changed rows keep the existing
source-index hint, bit comparison, lifetime and history handling. Dirty flags
live in a compact array beside the node records, and sparse traversal seeds stop
at the dense threshold because dense traversal visits the whole parent-first
order.

Both stores keep one private array per field group, all sharing one node index
space, so each pass streams only the bytes it reads. The hierarchy separates
source identity keys, cached local poses, 16-byte traversal/validity links,
matrices and revisions; the presented store separates traversal flags,
matrices and revisions. Interpolation history keeps a 64-byte hot interval
(previous pose, epoch and validity) apart from cold identity metadata. A node's
history hint and the record's node index form a pair that each topology rebuild
rewrites; a mismatch falls back to the full-Entity membership index. Motion that
leaves source identity keys unchanged skips the cold identity comparison. These
are storage layouts only: values, revisions and public reads are unchanged.

## Boundaries and compatibility

Authoritative reads observe the most recent transform propagation boundary.
Within a gameplay schedule, a local mutation does not automatically refresh its
world matrix; dependent systems retain their explicit ordering/boundary duties.
`Engine::refresh_transforms` and direct `propagate_transforms` include out-of-band
edits without advancing gameplay clocks.

Presented storage retains the last actual extraction result. An authoritative
refresh, inspector or collision query does not replace a fractional presentation
with current matrices. Interpolation reset/teleport, invalid branches, local
history lifetime and parent-before-child composition retain their contracts.
Current extraction and ordinary paused captures remain exact; a fractional
preview never changes authoritative simulation poses.

Collision reads authoritative `GlobalTransforms` through a shared resource
access, preserving its nonexclusive scheduler contract. It does not inspect
presented matrices. Game clients and development tooling use the same stores.
The agent's entity projections keep the `GlobalTransform` and
`PresentedTransform` names and JSON schemas; before preparation, transformed
entities continue to expose invalid/null presented values.

Rust source compatibility changes deliberately: derived component queries and
`Changed<GlobalTransform>` no longer compile. Resource reads, named boundaries
and retained output revisions replace those uses. See the
[migration](migrations/0.1-canonical-poses.md). Scene versions, saved IDs and agent
mutation/revision rules do not change.

## Retained render extraction

`Engine::extract_frame(aspect, mode)` prepares a `RenderFrame` borrowed from
retained core storage. Its immutable snapshot contains stable slots and source
ownership; visible indices select submitted slots. A frame identity and sequence,
structure/pose/color revisions and per-slot change stamps let consumers reuse
unchanged work. The borrow prevents gameplay edits while the frame is consumed.
Asset metadata resolves when source component stamps or the asset library's
identity/mutation revision changes. Pose hints validate full entity identity;
culling reuse depends on pose and view changes. `extract_frame` prepares
presentation from the just-refreshed hierarchy immediately before this pass, and
every admitted display pose is valid exactly when its authoritative pose is.
The pass therefore reads only the presented store, which still applies the
authoritative validity gate; debug builds assert that equivalence.

Retained frames do not copy displayed matrices into their instance records.
Each active slot keeps the presented-store index validated against its entity
during this pass, and `RenderFrame::model` reads the prepared pose through it
while the frame borrow prevents another preparation. When a slot's pose stamp
advances, the same pass also writes `RenderFrame::pose_rows()`: the transposed
top three rows (`PoseRows`, 48 bytes) of the matrix it already read for culling.
This is the renderer's upload layout, so changed slots upload directly from
core's rows without a second pass that re-reads and repacks every matrix; the
renderer keeps no CPU copy of poses. Retained `RenderInstance::model` fields are
unspecified, and an inactive slot's model and rows are arbitrary finite values
that are never submitted. Public owned snapshots still materialize models.

Per-slot extraction state is split likewise: a 32-byte record of pose revision,
resolved mesh/texture source indices, source change ticks and culling result is
read every frame, while slot ownership is read only on admission and removal.
Mesh bounds come from the resolved mesh source. A per-frame bit mask replaces
per-slot visit stamps for removal detection, and freeing a slot clears the
query-row cache entries that named it.

`source_revision` separates slot ownership, mesh/texture keys and source arrays
from visibility changes. The renderer retains a bounded slot-to-bucket mapping
under that revision. Visibility changes regroup compact IDs using cached buckets;
a previously hidden slot resolves its bucket on first submission. Public mutable
snapshots always use the independent validation path.

The renderer keeps GPU pose and color records at those stable slot indices.
Affine poses use three 16-byte rows (48 bytes per slot); colors use a separate
16-byte record. A grouped stream of 4-byte slot IDs selects visible instances
for each mesh/texture draw. Visibility and grouping changes therefore update
selection independently of pose/color storage. Metadata changes alone cannot
require color uploads. New slots, buffer replacement and an uncommitted upload
retry still require initialization; upload range coalescing may include unchanged
neighboring slots. Inactive slots remain bounded by the core slot limit and are
never submitted solely because their GPU storage exists.

The browser consumes this retained path. Public `extract` and
`extract_with_presentation` continue to produce independent, owned snapshots and
include out-of-band edits. Callers cannot use a mutable owned snapshot to forge
the retained frame's change provenance. Renderer upload storage and GPU handles
remain renderer-owned. Public snapshots retain their validated 80-byte instance
upload path; the compact affine representation is reserved for trusted core
frames.

Update (2026-10-03): no browser path drew owned snapshots, so the renderer
stopped accepting them. `extract` and `extract_with_presentation` now copy a
CPU-prepared frame: its visible slots with their displayed models, its joints,
decals, emitters and probes, and its local lights in world space. Meshes with
joints therefore never cull in owned snapshots, as in CPU-prepared frames. The 80-byte instance
records, their exact-byte upload cache, their vertex entry points, the
full-matrix shadow casters and core's separate owned extraction pass and
visibility cache were removed. Slot storage is now bounded by what one storage
binding holds of 48-byte poses rather than of 80-byte records.

[Decision 014](adr-014-gpu-presentation-and-culling.md) adds GPU-presented
retained frames for ordinary interpolated browser rendering. They keep these
slots, colors and pose storage, but list every candidate, leave `model()` and
`pose_rows()` unprepared, and let compute passes write poses and select visible
slots. Presented poses then stay in the CPU store until prepared on demand.

## Bounded interpolation blocks

Dense presentation gathers up to four independent previous/current local poses
into component-major SIMD lanes on the stack. Translation, scale and ordinary
small-angle quaternion blends run across those four poses. Larger rotations and
threshold neighbors retain the scalar rotation policy. Exact endpoints remain
exact. The four local matrices are then built in the same lanes, performing the
scalar builder's operations per lane, and transposed once; they are
bit-identical to building each blended pose's matrix. A lane-wise bit comparison
replaces converting blended poses back to transforms. The caller still applies
inherited reset/fallback decisions and composes world matrices in parent-first
order after blending.

Authoritative evaluation keeps building one local matrix per node. Batching
that construction required transposing already stored poses into lanes and
measured slower natively than the per-node path.

These blocks add no heap allocation or persistent second pose store. They are
a numeric kernel over canonical storage, not a rewrite of all transform storage
into a structure-of-arrays layout. Sparse presentation keeps its existing path.
Scalar and packed interpolation share the numerical policy and error budget;
they do not promise bit-identical results across every math backend. Sparse
composition visits affected branches, followed by a contiguous scan of admitted
nodes to retain a complete last-prepared snapshot. That scan does not publish
derived ECS components.

## Performance interpretation

This design removes duplicate derived-component maintenance and retains render
work across frames. It does not remove local change detection, matrix validation,
hierarchy dependencies or the authoritative workload. Browser measurements must
establish its benefit; native checks and fewer passes alone are not an FPS claim.
Preserve candidate/submitted counts, exact/fractional presentation, collision
behavior and capture identity in comparisons. No worker execution, lowered
simulation frequency or reduced entity population is implied.
