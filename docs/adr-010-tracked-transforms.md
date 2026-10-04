# Decision 010: tracked transform replacements

Status: superseded by [decision 011](adr-011-in-place-transforms.md). This document
records the former replacement-based API and its costs; use decision 011 for the
current mutable-query contract. `TransformEdits` itself was removed on 2026-10-03:
nothing used it but its tests. Extends decisions 006 and 008. Core owns mutation tracking,
propagation and the gameplay edit helpers. Bevy still owns the world, component
lifecycle and schedule execution; `Engine::app` remains the ordinary Bevy app.

## SDK migration

Stored `Transform`, `Parent`, and `GlobalTransform` are immutable Bevy components.
Typed `get_mut` and `Query<&mut ...>` access to these three types no longer compiles.
This is an intentional SDK change. Owned `Transform` values remain `Copy` with
public translation, rotation and scale fields; scene JSON does not change.
Gameplay replaces local poses and uses checked `set_parent`/`set_parents` for
relationships. Derived globals remain engine-owned.

Read local poses and queue edited copies through `TransformEdits`:

```rust
use engine::prelude::*;

#[derive(Component)]
struct Moving;

fn move_objects(
    time: Res<FixedTime>,
    poses: Query<(Entity, &Transform), With<Moving>>,
    mut edits: TransformEdits,
) {
    for (entity, pose) in &poses {
        edits.edit(entity, pose, |next| {
            next.translation.x += time.delta_seconds;
        }).expect("bound Moving entities by MAX_TRANSFORM_EDITS");
    }
}
```

`edit` skips bit-identical results; `set(entity, pose)` queues an owned replacement.
Each system retains one buffer, limited to 100,000 queued replacements between
deferred-application points. Handle `CapacityExceeded` explicitly. Previously
accepted edits remain queued; `set_batch` rolls back its whole new batch on
overflow. Targets missing a local transform when application begins are skipped.
Admission checks entity generations and component membership, reusing the result
within each archetype run instead of fetching every old pose a second time.
Replacements apply in submission order. Two edits based on the same old pose
overwrite rather than combine; use one local copy or an ordered read after apply.

Edits become visible at Bevy deferred sync points and schedule ends. Chain
dependent systems with `.chain()` or explicit ordering so later readers see
earlier local replacements. Derived globals still refresh at the existing engine
boundaries; chaining local writers/readers does not itself propagate globals.

For an immediate exclusive-world edit:

```rust
edit_transform(world, entity, |pose| pose.translation.x += 1.0)?;
```

The helper returns `MissingEntity` or `MissingTransform`, skips unchanged values,
and preserves the existing component's added tick. After a teleport, call
`reset_interpolation`; refresh transforms before inspecting the derived matrix.
Direct copy/edit/`world.entity_mut(entity).insert(pose)` also preserves the added
tick. Bevy's `World::modify_component` is tracked, but removes/reinserts an immutable
component and resets its lifetime. Use replacement or the helpers for ordinary
motion so interpolation retains its interval.

## Tracking and bounded fallback

Lifecycle hooks observe ordinary `World` and `Commands` insertion, replacement,
removal and despawn. Tracking does not depend on a plugin declaring itself safe,
Bevy change ticks advancing, or callers using the convenience helpers. Plugins,
run conditions, custom executors and transform schedule extensions keep their
existing execution semantics; no app wrapper or schedule skipping is introduced.
Local and global pose tracking uses add/discard/remove hooks: discard records a
replacement before its old value disappears, so a second insert notification is
unnecessary. Bevy's user insert observers still run normally.

A journal holds at most 100,000 notifications per buffer. Sparse propagation
validates notified entities, deduplicates repeated visits, and recomputes affected
branches. Unchanged boundaries scan no transform entities. First use, structural
changes, journal overflow, and populations at/above the 100,000-node admission
limit use authoritative full scans. Notification counts reaching 75% of cached
sources also select a contiguous scan. This threshold is a provisional policy;
it does not establish the fastest choice for every browser or workload.

Overflow requests full validation; it never silently drops an edit. Existing
cycle, orphan, depth, matrix-validity and capacity behavior remains. Observers
run normally during replacements. Any further notifications produced during
propagation are consumed before it returns fresh data. Feedback must settle in
eight passes; otherwise propagation panics explicitly instead of looping forever.

Derived evaluation collects changed node indices, then installs globals through
Bevy batches of at most 1,024 rows. One cached-query read per candidate preserves
no-op detection and does not recreate removed globals. Bundle/archetype setup and
suppression setup are shared across a batch; source validation likewise updates
query metadata once before reading notified entities.

The engine suppresses only an ordered batch's exact expected entity/old-value
discard notifications. Suppression clears on both success and unwinding. Value
IDs remain private; copied globals cannot alias different matrices. Observer
commands flush after all derived batches, with suppression cleared, and their
edits settle before propagation returns. Observers should not assume that each
derived insertion immediately flushes its commands or that all derived globals
are fresh while insertion is in progress. Local edit batches also flush observer
commands before the next ordered system. A caught observer panic forces a full
cache rebuild on the next propagation boundary.

Batch application and tracking are separately profiled (`apply_transform_edits`
and `tracked_validation`), alongside fallback scans and dirty propagation.
Dirty propagation includes separately measured evaluation and global writes;
observer-command flushing has its own scope. Global writes retain at most one
node index per admitted transform plus one 1,024-row payload/identity batch.
Queue buffers, journal storage and caches retain bounded capacity. These contracts
make costs visible; performance claims require matched browser captures.
