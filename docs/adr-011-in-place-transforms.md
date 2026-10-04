# Decision 011: in-place transforms and shared interpolation capture

Status: adopted. Supersedes decision 010's immutable-pose mutation contract and
extends decisions 006 and 008. Core owns the implementation; the public app,
world and schedules remain ordinary Bevy APIs.

[Decision 013](adr-013-canonical-poses-and-retained-rendering.md) supersedes this
record's derived `GlobalTransform` component, repair/write paths and
`Changed<GlobalTransform>` observation contract. Local `Transform` mutation,
change detection and shared history capture continue to apply.

## Why change the previous design

The 90,000-mover browser capture after replacement batching still spent about
8.9 ms per simulation tick in local and derived write paths, plus 3.6 ms copying
interpolation histories. A sampled Chrome profile independently identified
replacement/lifecycle work, hierarchy processing and history capture as major
costs. Profiling overhead and compiler flags did not explain the bottleneck.

Immutable components guaranteed that every edit emitted a lifecycle notification,
but made ordinary numerical updates pay replacement, hook and repeated lookup
costs. Batching reduced setup without removing that per-entity work. The new
design streams ordinary mutable updates and shares source observations with
interpolation, rather than maintaining another full history capture pass.

## Gameplay API and observer semantics

`Transform` and `GlobalTransform` are mutable Bevy components. `Parent` remains
immutable; use checked `set_parent` / `set_parents` for topology edits. Globals
remain derived engine-owned values; gameplay normally reads them.

```rust
fn move_objects(time: Res<FixedTime>, mut poses: Query<&mut Transform, With<Moving>>) {
    for mut pose in &mut poses {
        pose.translation.x += time.delta_seconds;
    }
}
```

Ordinary `World::get_mut`, mutable queries, `modify_component`, and component
replacement are observed at the next transform propagation boundary. No game
plugin registration, manual dirty flag or trusted-system declaration is needed.
`edit_transform` edits a copy, skips bit-identical results and commits in place.
Decision 010's deferred `TransformEdits` was removed on 2026-10-03: only its tests
used it, and its fixed replacement cap broke decision 029's no-caps rule.

In-place edits and engine global updates follow Bevy's normal mutation semantics:
they set change stamps and do **not** fire Insert/Discard lifecycle observers.
Explicit insertions/replacements still fire those observers. Use `Changed<T>`
or the engine's named schedule boundaries to observe numerical updates. This is
an intentional change from decision 010's replacement-based engine writes.

`bypass_change_detection`, manually rewriting change stamps, and restoring a
World's tick counter opt out of the tracking mechanism and are unsupported for
transform updates. Ordinary `clear_trackers` is supported. This explicit opt-out
is the same distinction used by Bevy's mutation APIs; it is not a permission to
silently bypass validation in an engine system.

## Tracking and bounded work

The hierarchy owns a `SystemState` change window and retains its last successful
scan tick for age checks, independently of the World's last-change tick and
gameplay systems. Propagation flushes pending commands and performs tick
maintenance. On ordinary boundaries, `Or<(Changed<Transform>,
Changed<GlobalTransform>)>` tests metadata before fetching component tuples.
Unchanged rows still require timestamp checks, but do not load source records or
compare pose payloads. Changed rows resolve cached sources by full generational
entity identity, with a checked consecutive-index hint before the map lookup.
Unrelated archetype moves therefore need not rebuild the cache. Private global
value IDs still distinguish copied derived values and avoid unnecessary matrix
comparisons.

After a successful boundary the filter window acknowledges all engine-derived
global writes, then the World tick advances. Those writes do not appear as new
external edits on the next scan, while a subsequent ordinary mutable write is
newer even when no schedule or gameplay tick has run between two refreshes.
Wrapping-aware comparisons, an aged-window fallback and Bevy's CheckChangeTicks
notification handle stamp aging. A caught panic leaves the cache incomplete and
forces authoritative reconstruction on its next use, even if fetching the query
already advanced its internal window.

Structural hooks invalidate topology and interpolation membership. Topology
rebuilds remain bounded to the transform-node limit (now 262,144), validate cycles/depth/orphans,
and retain the existing invalid-matrix and admission rules. First use, structural
changes, interrupted propagation, aging and populations at or above the admission
limit use an unfiltered source scan. Dense propagation
walks parent-first order; sparse changes follow affected branches. Dense global
writes stream the mutable ECS column; sparse writes resolve only queued nodes.
No replacement hooks, suppression journal or matrix payload batch is needed for
engine-derived updates. Bevy change stamps are set only when a value changes.

This trades zero-work unchanged boundaries for a lightweight metadata sweep.
It removes lifecycle processing from every numerical change. The sweep and
dense/sparse policies are measured costs, not a universal performance claim.

## Matrix validation

The matrix admission predicate also avoids constructing a general inverse when
conservative coefficient and determinant bounds prove that the existing inverse
would be finite. Outside those bounds it runs the original inverse check.
`hierarchy/validation.rs` documents the proof against pinned glam 0.30.9 and
compares both paths over ordinary, boundary and extreme inputs. Revisit that
proof when upgrading the inverse implementation. Affine, singularity and normal
conditioning rules remain unchanged.

## Interpolation ownership

Hierarchy source admission also observes opt-in marker/lifetime/parent changes.
During a fixed interval, the first source change captures its previous endpoint;
later changes update the hierarchy's current source pose without overwriting
that previous endpoint. Histories borrow the current pose from `Source.local`
during presentation instead of retaining a duplicate current transform.
Tick begin/end use epochs rather than full history queries. Structural changes
reconcile membership, while validity transitions and explicit resets retain
their required cancellation behavior. Cached slots are validated by full Entity.

Presentation consumes the completed interval. Out-of-tick edits snap, unchanged
entities do not replay an earlier interval, and unmarked descendants still
inherit interpolated parent motion. Exact-current rendering, teleport resets,
invalid ancestry and conservative numerical fallback retain their contracts.
This decision did not change quaternion interpolation or simulation frequency.
[Decision 012](adr-012-transform-validation-and-presentation.md) subsequently
adds the small-angle presentation policy; simulation frequency remains unchanged.

All scratch storage is bounded and reported. Histories retain the previous
endpoint and interval metadata; the hierarchy owns the sole cached current pose.
Epoch metadata and source callbacks avoid repeated full history capture passes.

## Research and validation

Bevy's [component mutability contract](https://docs.rs/bevy_ecs/latest/bevy_ecs/component/trait.ComponentMutability.html)
distinguishes mutation from replacement hooks. Its [change-filter documentation](https://docs.rs/bevy_ecs/0.19.1/bevy_ecs/query/struct.Changed.html)
documents metadata filtering as O(N), and [change-detection APIs](https://docs.rs/bevy_ecs/latest/src/bevy_ecs/change_detection/traits.rs.html)
make bypasses explicit. These support the design tradeoff; they do not establish
our engine's speed. Performance conclusions require matched browser workloads.

Focused checks cover ordinary direct/query edits, tracker cleanup, same-clock
writes, aging/wraparound, copied globals, structural reorder, explicit observer
commands, panic recovery, interpolation endpoints/resets and invalid matrices.
The 25% and 100% stress configurations remain the practical browser checks.
