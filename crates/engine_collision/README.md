# engine_collision

Owner: engine maintainers. This crate provides authoritative spatial queries and
fixed-tick sensor transitions. It depends on `engine_core`, Bevy and pinned Parry
0.26.1; core does not import collision. It has no browser, GPU, worker, serialization
or rigid-body solver dependency. Execution stays on the owning engine thread.

## Entry points and scheduling

Add the unit `CollisionPlugin` before the first simulation frame. It installs
`CollisionWorld`, `CollisionConfig` and `SensorEvents`. Collider data is usable
without the plugin, but `refresh_collision_world` then returns `MissingPlugin`.
Games without the plugin do no collision scans. Empty core pre/post schedules add
no transform propagation passes.

`FixedPreSimulation` synchronizes the collider snapshot from authoritative globals.
Gameplay systems query it in `FixedSimulation`; it remains a snapshot while those
systems run. Deferred commands apply, globals propagate, and `FixedPostSimulation`
runs `CollisionSystems::Sync` followed by `CollisionSystems::Detect`. Register
sensor consumers in that post schedule with `.after(CollisionSystems::Detect)`.
Post commands apply before the final propagation and interpolation-history capture.
Changes made by post consumers reach the collision cache at the next refresh.

Scheduled sync uses read-only collider/transform queries and writes only collision
resources; detection writes the snapshot and sensor events. Neither requires
exclusive world access. The explicit refresh API lazily retains query metadata
separately from the snapshot, so replacing `CollisionWorld` or adding archetypes
does not reuse stale snapshot data.

Call `refresh_collision_world(&mut World)` for fresh queries after paused direct
edits or explicit same-phase gameplay edits. It propagates authoritative transforms
and refreshes query storage; it does not advance clocks, prepare display matrices,
commit sensor transitions or consume events. Rendering and interpolation never
refresh collision implicitly. Inspection can therefore refresh collision without
altering the last rendered pose or fabricating gameplay events.

## Collider geometry and numeric domain

Attach checked `Collider::cuboid(half_extents)` or `Collider::sphere(radius)` values
alongside `Transform`. `ColliderShape`, `shape()`, `layers()` and `is_sensor()` are
the public data contract. `with_layers` and `with_sensor` construct replacements.
Colliders need neither a renderable nor a mesh, and render mesh bounds do not
define their shape. `CollisionLayers { memberships, filter }` uses symmetric
filtering: each participant's filter must include the other's membership bits.
Default membership is bit zero; the default filter accepts every bit.

The complete authoritative matrix from `GlobalTransforms` applies to the local
primitive. Scheduled synchronization reads this resource alongside collider/local
components; it does not query a derived ECS matrix column or use presented poses.
Hierarchical translation, rotation, reflection, nonuniform scale and shear remain
intact: a stretched sphere is an ellipsoid, and a sheared box a parallelepiped.
No transform decomposition, vertex hull allocation or presented pose is used.

Local half extents/radius must be finite and within `[1e-4, 1e4]`. The world
transform must pass core's affine validation; its center coordinates must lie
within `[-1e6, 1e6]`. Each shape-scaled basis column length must lie in
`[1e-4, 1e4]`. After dividing the basis by its longest column, its inverse infinity
norm must be at most 1000. `validate_collider_transform` is shared with scene
validation. This collision envelope is narrower than render admission and is a
numerical operating limit, not a world-unit scaling recommendation.

Rays use finite origins and endpoints within the coordinate bound and distances
in `[0, 2e6]`; nonzero directions are normalized internally. Shape casts validate
both start and translated end center. Invalid data, orphaned hierarchies and
capacity overflow appear in `CollisionReport`. An incomplete snapshot rejects
queries explicitly, so a failed collision update cannot masquerade as empty space.

## Queries

`CollisionWorld::raycast` returns the nearest `RayHit` or `None`. `raycast_all`
fills a caller-owned reusable vector sorted by distance then entity identity.
Distances are world units. Ray normals point out of the target. A ray beginning
inside or on the boundary reports distance zero, `started_inside: true`, and a
zero normal because there is no unique entry surface.

`overlap(shape, matrix, filter, output, limit)` returns sorted generational entity
IDs after narrow-phase testing. `QueryFilter` combines symmetric layers,
`SensorFilter::{Include, Exclude, Only}`, and one excluded entity. Limits must fit
the configured hit bound. Candidate or result exhaustion returns an error;
vector outputs are cleared. Nearest queries also fail on candidate exhaustion
rather than claiming their partial best hit is globally nearest.

`cast_shape(shape, matrix, displacement, filter, options)` sweeps translation with
fixed orientation/scale against the snapshot. It returns the nearest fraction
in `[0,1]`; it does not apply motion or solve contacts. The normal points out of
the target. `ShapeHit::point` is a representative target surface witness, not a
contact manifold or guaranteed closest point under shear. With
`stop_at_penetration: true`, initial contact blocks at zero. With false, initial
tangent/separating movement is allowed. `started_overlapping` includes initial
touching and contact within the configured skin or numerical tolerance.
`target_distance` is a world-space skin in `[0,1]`; for two cuboids it expands SAT
intervals conservatively, so corners do not represent an exact Euclidean rounded
hull. Zero skin is used by the sample character's axis casts.

Cuboid pairs use 15 separating-axis families in f64, including face normals and
edge cross products. The same intervals support translation casts and handle
exact contact on wide floors without unstable iterative normals. Other convex
pairs use Parry's support-map GJK/cast routines, with a retained EPA workspace for
initial penetration normals. Overlap tests (sensors and `overlap` queries) first
settle pairs far from contact with conservative f64 pre-tests that keep every
decision: overlapping inscribed balls, or reaches along the line between the
centers that cover it, prove contact; for pairs that would use GJK, disjoint
bounding balls or a separating axis along that line prove separation. Settled
pairs skip the exact test, so a GJK support-call exhaustion on such a pair no
longer surfaces as `NumericalFailure`. Box pairs left over stop at their first
separating axis and only normalize axes whose unnormalized gap is positive. Rays transform into the unit shape in f64 and use
analytic box/ellipsoid intersections. These algorithms have floating-point
tolerances; there is no cross-platform bitwise determinism guarantee. Invalid
cast geometry and detected support-call exhaustion return `NumericalFailure`.
Parry also has numerical-stagnation exits indistinguishable from a miss; the
numeric domain and regression tests reduce risk but do not prove exact collision
classification for every boundary configuration.

## Sensor lifetime and budgets

Only pairs containing a sensor generate automatic events. Solid-solid overlap is
available through queries; sensors apply no forces or movement response. Detection
uses post-simulation poses, so crossing a sensor entirely between ticks can be
missed; use shape casts when continuous detection is required.

`SensorEvents` holds the latest attempted fixed tick, a monotonic sequence, the
last complete tick, canonical active pairs, and `Enter`/`Exit` events. Read them in
the post schedule for every tick when catching up. Repeated reads do not consume
them. Despawn, generation reuse, collider removal and layer changes produce exits
at the next complete detection boundary. Events may reference despawned entities;
check existence before reading their components. Entity IDs are not save IDs.

Dense overlapping scenes inherently have quadratic possible pair counts. Candidate
visits and retained pairs are separately bounded. An incomplete pass clears its
event list, preserves the previous complete pair set and exposes an error; it
does not emit false exits for unvisited pairs. A later complete pass computes the
transition from that retained set. Transient overlaps during failed ticks cannot
be recovered and are not claimed as complete observations.

| Budget | Default | Hard maximum |
| --- | ---: | ---: |
| Collider records | 10,000 | 100,000 |
| Sensor candidate pairs per tick | 100,000 | 1,000,000 |
| Active sensor pairs | 16,384 | 65,536 |
| Query candidates per call | 10,000 | 100,000 |
| Query result hits | 1,024 | 4,096 |

Zero budgets are valid. Invalid configurations fail explicitly. Changing
`CollisionConfig` takes effect on refresh. Candidate work is charged before layer
filtering so rejected filters cannot hide unbounded work.

## Ownership and costs

The collider query is cached. Unchanged refresh scans at most the admitted inputs,
compares each entity and collider plus the authoritative pose's output revision
(read through a validated `TransformIndex` hint kept with the input, never the
uncacheable `u64::MAX` stamp) and preserves the cache revision. Equal revisions
certify equal matrix bits and validity, so no matrix is compared or copied.
Changed refresh rewrites inputs in place. With unchanged membership and statuses
it refits the balanced flat AABB BVH bottom-up in O(N): internal bounds stay exact
unions of their children, so traversal visits the same leaves and queries (which
sort or break ties by identity) return what a rebuilt tree would. Otherwise, or
once refits grow the summed internal surface area past 1.5 times the last build,
it rebuilds in O(N log N) using retained vectors; the reused entity-status map is
rebuilt only when membership or a status changed. The tree has at most `2N-1` nodes; median splitting keeps
build recursion and query stacks logarithmic (the bounded query stack has 32
slots). Broadphase queries prune whole branches; the 10,000-collider sparse test
visits fewer than 100 tree nodes for a local overlap. There is no full ECS scan
inside a query and no collision work in render extraction.

Input, entry, bounds, BVH, pair and event buffers retain capacity. First use or a
new population/contact-complexity peak can allocate; steady workloads reuse those
buffers. Parry's pinned EPA resets and reuses its private workspace, with a fixed
iteration bound. It is called only when penetration geometry is needed. Caller
result vectors similarly grow to their peak and should be retained.
`CollisionReport::cache_bytes` counts owned input/entry/bounds/BVH Vec capacities;
it excludes HashMaps, pair/event vectors, Parry workspace, ECS metadata, allocator
overhead and process heaps. It is not a total memory estimate.

Native tests cover geometry, shear/reflection, numerical admission, exact-contact
movement, filtering, bounded outputs, sparse broadphase work, unchanged storage,
sensor sequencing, overflow recovery, generations and authoritative/presented
separation. A deterministic 1,000-case box comparison checks SAT results against
Parry's independent cuboid query. Browser workloads remain the source for browser
performance claims.
