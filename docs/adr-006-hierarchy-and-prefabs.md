# Decision 006: local transforms and reusable scene prefabs

Status: adopted. Extends decisions 002 and 005. Core owns runtime transforms;
scene owns text expansion and instance disposal; development tooling projects
the same data. The renderer continues consuming owned world matrices and asset
handles. This feature adds no browser calls or string lookup to gameplay loops.

[Decision 011](adr-011-in-place-transforms.md) supersedes decision 010 and the
replacement-journal implementation notes below. `Transform` and `GlobalTransform`
are ordinary mutable components; frequent gameplay motion uses mutable local
queries while globals remain engine-owned. A private change-tick watermark
drives source validation, and propagation updates globals without insertion
events. `Parent` remains immutable with checked edits; `edit_transform` writes in
place. Decision 011
also defines change-detection opt-outs and fused interpolation source capture.

## Runtime contract

[Decision 013](adr-013-canonical-poses-and-retained-rendering.md) supersedes the
derived-component storage and observation details in this historical record.
Read authoritative poses from `GlobalTransforms`; local mutation and checked
parent/topology semantics below remain applicable.

`Transform` is local TRS. `Parent` is a non-owning generational entity reference;
use checked `set_parent` or atomic `set_parents` at an exclusive world boundary.
Changes preserve the local transform. They reject missing entities/transforms,
cycles and moved subtrees exceeding 64 parent edges. Detachment is explicit and
does not preserve the previous world pose automatically.

`GlobalTransform` is a derived full matrix, not a decomposed TRS. Multiplication
preserves shear when a rotated child inherits nonuniform parent scale. Gameplay
edits local transforms; propagation writes globals. Invalid/nonfinite/singular
matrices, orphans and over-budget branches have no valid global and skip render
extraction. `RenderSnapshot.invalid_transforms` counts skipped renderables; the
browser exposes the same counter. Healthy branches remain renderable. Normal
calculations rescale the basis/cofactors and vector before normalization. Admission
limits the inverse infinity norm of the coefficient-normalized normal matrix to
10,000, rejecting severe anisotropy/near-singular shear in every scene version.
This tightens previous numerical admission; see the migration guide.

Core refreshes transforms before simulation, after each fixed tick's deferred
commands, after presentation, and before extraction. Systems within a schedule
read the previous boundary's globals. Use `Engine::refresh_transforms` or
`propagate_transforms` for an explicit exclusive boundary after direct edits.
Paused inspection refreshes derived transforms without advancing gameplay.

Propagation admits 262,144 transform nodes (100,000 until 2026-09-25, raised with
the engine's render-instance limit for larger workloads) and reuses bounded vectors and an
entity index map. Tracked replacements, including same-tick ECS edits, identify
sources to validate; first-use, structural, dense and overflow paths retain the
full input scan described in decision 010. Local movement retains cached topology,
traversal order and child links; matrix work follows changed nodes and affected
descendants, and unchanged globals are not rewritten. Structural/query-order
changes rebuild the topology cache. Cycle/orphan/depth validity is independent
of pose validity, allowing repaired matrices to restore otherwise valid branches.
Relationship edits still validate atomically outside ordinary propagation.
No scheduling boundaries, ownership rules or validation limits change.
Profiling distinguishes topology rebuilds from dirty propagation and counts
evaluated/written nodes. Browser/device gains require matched measurements.

Parenting never transfers ownership or cascades despawns. Disposing a scene
removes its own entities/assets only. Children owned elsewhere become orphans
until explicitly detached or reparented. Handles copied outside an owner remain
weak. Cameras remain a world-space resource in this milestone.

## Scene version 3

Versions 1 and 2 keep their original flat semantics. Version 3 adds optional
`parent`, optional paired `mesh`/`material`, named `prefabs`, and optional
`prefab` on a transform-only placement root. Definitions reference scene assets.

```json
{
  "version": 3,
  "id": "example",
  "revision": 1,
  "meshes": [{"id":"box","source":{"kind":"cube"}}],
  "materials": [{"id":"teal","color":[0.1,0.8,0.6,1]}],
  "prefabs": [{"id":"lamp","entities":[
    {"id":"base","mesh":"box","material":"teal"},
    {"id":"cap","parent":"base","mesh":"box","material":"teal",
     "transform":{"translation":[0,1,0],"scale":[0.5,0.5,0.5]}}
  ]}],
  "entities": [
    {"id":"west","prefab":"lamp","transform":{"translation":[-4,0,0]}},
    {"id":"east","prefab":"lamp","transform":{"translation":[4,0,0]}}
  ]
}
```

One placement creates its root and stable qualified children: `west.base` and
`west.cap`. Definition roots attach to the placement root; local parent IDs
resolve within the definition. Expanded IDs may not collide with authored or
other expanded IDs and must fit the existing 64-character identifier bound.
Changing a placement moves its subtree. Changing a definition affects all its
placements on the next reload. Per-placement child overrides and nested prefabs
are deliberately unsupported.

The source limit remains 1 MiB, with at most 128 definitions, 10,000 total
definition objects and 10,000 expanded runtime objects. Expansion validates
all references, depth, cycles and composed matrices before allocation. Failed
admission rolls back newly allocated assets/entities. Replacement stages a
validated instance before disposing the previous one.

Meshes, materials and textures are allocated once per scene instance, regardless
of prefab placement count. Loading two independent `SceneInstance` values still
creates independent owners and asset handles. Prefabs do not add a hidden global
cache or alter explicit disposal. Metadata retains definition/instance/local IDs;
games attach typed components from tags once at startup, including expanded
children, rather than looking up authoring strings every frame.

`asset import --as-prefab` creates one definition with no placement. Native GLB
node transforms are still baked into imported geometry. This mode groups the
resulting static parts; it does not reconstruct the GLB node hierarchy or add
animation. `--replace --as-prefab` replaces that definition and its generated
assets while preserving placement records, then validates the complete candidate
before the shared revision-checked save.

## Agent interface

`describe` exposes generated version 3 schemas plus `Hierarchy` and
`GlobalTransform` projections. `inspect` with `parent` pages current direct
children; it is mutually exclusive with `entity` and cannot select resources.
The same 10,000-record scan and 100-row response limits apply.

`Hierarchy` distinguishes source and current parents and reports prefab identity.
`parent_in_active_scene` indicates whether its parent ID resolves to that exact
runtime entity in the inspected instance, avoiding ambiguity between independent
instances of the same source. `GlobalTransform` returns a column-major matrix
and translation or `valid:false` with null values. It does not invent a TRS
decomposition for sheared matrices. Source changes still use validate/save/reload;
inspection does not serialize runtime entity indices as persistent IDs.

Crystal Garden demonstrates two lantern placements sharing assets and a crystal
accessory parented to the player. Its small collision systems explicitly require
root renderables for player/solid/pickup tags. General hierarchy-aware physics,
visibility culling, interpolation and nested prefab overrides remain separate work.
