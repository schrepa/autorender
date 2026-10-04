# Decision 009: collision geometry and spatial queries

Status: adopted. Collision is an opt-in subsystem in `engine_collision`, exposed
through `engine::collision` and the gameplay prelude. It depends on core; core
does not import it. Rendering does not own collider geometry or visibility.

## Scope and ownership

`Collider` stores a checked cuboid or sphere, symmetric membership/filter masks,
and a sensor flag. Attach it alongside `Transform`. A collider does not require a
mesh. Its authoritative `GlobalTransform` applies the complete affine hierarchy:
rotated and sheared boxes stay convex boxes; stretched spheres become ellipsoids.
Offsets use ordinary child transforms. Presented/interpolated poses never enter
collision queries.

`CollisionPlugin` installs a bounded `CollisionWorld`, its configuration, and
fixed scheduling systems. Games without the plugin incur no collision scans.
Collider components remain valid data without the plugin; query refresh reports
the missing subsystem. Plugin registration follows Bevy's unique-plugin rule;
runtime plugin removal is unsupported.

The engine provides closest/all ray queries, overlaps, translation-only shape
casts, and sensor enter/exit observations. Shape casts keep orientation fixed
through the sweep. This milestone has no rigid-body integrator, impulse solver,
friction, restitution, joints, triangle-mesh colliders, or rotating continuous
collision detection. Gravity, sliding, grounding, damage, and pickup behavior
belong to games. A general character controller can be added separately.

## Scheduling and observation

Each fixed tick runs `FixedPreSimulation`, `FixedSimulation`, and
`FixedPostSimulation`. Authoritative transforms propagate at the boundaries.
Collision synchronizes before gameplay, then synchronizes again and detects
sensor transitions after gameplay commands. Game systems consume that tick's
sensor snapshot in `FixedPostSimulation`, ordered after
`CollisionSystems::Detect`. Post-simulation commands apply before final transform
propagation and interpolation history capture.

`CollisionWorld` queries use its last synchronized snapshot. Systems that mutate
transforms after synchronization must explicitly arrange another refresh before
querying those changes. `refresh_collision_world` refreshes authoritative
transforms and the spatial cache without advancing simulation or committing
events. Agent geometry queries use this boundary. They leave input, clocks,
presented poses, and the last sensor snapshot unchanged.

Events are a non-consuming snapshot of the latest attempted fixed tick, not an
unbounded queue. Multiple ticks in one step replace earlier snapshots. Sensor
pair updates commit atomically: if detection exceeds a budget, the previous
complete pair set remains, the attempted tick reports incomplete, and no false
exits are emitted from a partial scan. A later successful tick reconciles pairs.
Runtime entity references are generational; authored IDs are supplied separately
by scene ownership and are never reconstructed from entity indices.

## Geometry, bounds, and dependencies

The broad phase uses a retained flat AABB tree. Queries apply exclusion, symmetric
layer masks, and sensor selection before exact convex work. Nearest queries must
complete their eligible scan; exhausted budgets return errors, never a partial
answer presented as the nearest hit. Bounded overlap/all-hit results likewise
fail explicitly if their output limit is exceeded.

Ray tests use inverse affine transforms and inverse-transpose normals. Affine box pairs use bounded 15-axis separating-axis tests in f64,
including continuous translation intervals. Sphere/mixed convex overlap and
cast work uses pinned `parry3d` 0.26.1 through a private affine support
adapter; Parry types are not part of the SDK. This version shares our pinned glam
math line. Only required 3D/f32, standard-library and allocation features are
enabled. Its upstream license and attribution remain in the dependency graph.

Finite values alone are insufficient for numerical reliability. Shared authored
and runtime validation admits local/effective extents from 1e-4 through 1e4,
centers within ±1e6, valid invertible render transforms, and bounded basis
conditioning. Exact contact boundaries still use floating-point tolerances.
Runtime-invalid colliders and capacity overflow make the cache incomplete;
queries fail explicitly. Inspect the cache report before interpreting results.

Budgets bound admitted colliders, query candidates/results, detection candidates,
and committed sensor pairs. Buffers retain capacity. Cache construction and
narrow-phase work have CPU costs even when no collider is visible. This feature
does not claim to improve the earlier interpolation benchmark; browser
measurements belong in the verification record.

## Authoring and agent workflow

Scene version 4 adds optional collider data to objects, prefab template nodes,
and placement roots. Versions 1–3 still load with their original colliderless
semantics; their schemas are frozen. Complete source validation, including
expanded world transforms, precedes mutation. Reimport preserves authored
colliders for matching source IDs and rejects removal of collider-bearing IDs.

Agents discover `Collider`, `CollisionWorld`, and the `protocol` schema (`describe` with name `protocol`), then use
`raycast`, `overlap`, `shape_cast`, or paged `collision_events`. Query poses are
world position and quaternion; target colliders retain their full affine
transforms. Hits identify active authored entities when possible, and explicitly
mark runtime-only or removed entities. Requests and responses retain the shared
16 KiB/64 KiB limits. Persistent changes use the existing validated, revision-
checked scene save/reload path.

Crystal Garden authors its floor, walls, player volume, and pickup sensors. Its
Rust controller casts the player before movement, and its collection system
consumes sensor enters. Layer meanings and score remain game policy. The sample
requires collider-bearing gameplay tags; generic engine scenes need no colliders.

See the [migration](migrations/0.1-collision.md), [subsystem contract](../crates/engine_collision/README.md),
and [agent workflow](agent-workflow.md).

Upstream implementation reference: [Parry 0.26.1](https://github.com/dimforge/parry/tree/v0.26.1).
