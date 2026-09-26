# Architecture decision 008: opt-in render interpolation

Owner: `engine_core` owns pose history, presentation hierarchy and extraction;
the browser composition selects a presentation mode. Games select which local
transforms interpolate. GPU rendering consumes the resulting snapshot unchanged.

[Decision 013](adr-013-canonical-poses-and-retained-rendering.md) moves derived
pose reads into canonical resources. `PresentedTransform` is now a read value
from `PresentedTransforms`, not an ECS component. The interpolation and exact
capture semantics in this record remain unchanged.

[Decision 014](adr-014-gpu-presentation-and-culling.md) evaluates ordinary
interpolated browser frames on the GPU with the same policy. The CPU pass below
remains the reference for exact captures, `present {alpha}` previews and
inspection, which materializes a pending GPU-presented frame on demand.

## Gameplay and presentation

Add `InterpolateTransform` to an entity whose local `Transform` is updated by
`FixedSimulation`. The engine retains previous/current local poses at fixed-tick
boundaries. Translation and scale interpolate linearly; rotation follows the
short quaternion arc, using the small-angle policy below from
[decision 012](adr-012-transform-validation-and-presentation.md).
Interpolated local matrices compose through the parent
hierarchy, preserving shear from rotated nonuniform ancestors. An unmarked child
still follows its parent's presented pose. Mark a child when its own local
animation also needs interpolation.

Rotation endpoints are normalized before blending. When their absolute dot
product is at least `0.9995` (about 3.624 degrees apart), the engine uses
shortest-hemisphere normalized linear interpolation. Larger rotations retain
SLERP. This follows the small-angle policy used by
[Unity Mathematics](https://github.com/Unity-Technologies/Unity.Mathematics/blob/master/src/Unity.Mathematics/quaternion.cs)
and avoids SLERP's trigonometric work for ordinary small fixed-step turns.
It slightly changes angular speed: the ideal maximum angular deviation at this
threshold is approximately `1.015e-6` radians. Focused f64-reference checks use a
`1e-5`-radian budget including f32 rounding, across general orientations,
quaternion signs and accepted near-unit inputs. This is a deliberate numerical
policy change, not bit-equivalent output. Alpha zero still uses the exact previous
local pose and alpha one uses authoritative current matrices. Hierarchy validity
and numerical fallback still run on the composed result; translation and scale
arithmetic are unchanged.

`Transform` and `GlobalTransform` remain authoritative gameplay data. Rendering
does not write smoothed values back into them. `PresentedTransform` describes
the separate matrix prepared at the last extraction boundary. Culling and draw
records use that same matrix, avoiding camera-edge disagreement.

The ordinary browser frame uses the fixed clock's remaining fraction. Conventional
previous/current interpolation displays a timeline one fixed tick behind wall
time: approximately 16.7 ms at 60 Hz. It improves motion continuity between ticks;
it does not increase simulation frequency. Immediate presentation controls such
as the fly camera keep their existing per-frame behavior.

```rust
use engine::prelude::*;

engine.app.world_mut().spawn((
    Transform::default(),
    InterpolateTransform,
    Cube { color: [0.2, 0.8, 0.7, 1.0] },
));
```

This is runtime gameplay policy. Scene JSON stays at version 3; games can attach
the marker when instantiating their own tagged objects. Crystal Garden opts in
its player and animated pickups; the accessory follows its parent. Orbit opts
in its animated cubes. Static scenery does not opt in.

## Discontinuities and ownership

Use `reset_interpolation(world, entity)` after a teleport or other discontinuous
local change. `reset_all_interpolation(world)` snaps all retained histories;
`Engine::reset_clock` also resets it at suspension/resumption and successful scene
replacement. New entities start from their current pose. Generational identity
prevents a despawned object's history attaching to its replacement.

Topology changes, invalid transforms and scale-sign changes must not interpolate
through unrelated or singular poses. Changes outside fixed simulation, including
presentation-system writes, snap to current state. Gameplay should not write
`PresentedTransform`.

## Explicit extraction and agent controls

`Engine::extract` preserves its exact-current contract.
`Engine::extract_with_presentation(aspect, snapshot, mode)` selects
`PresentationMode::Current` or `PresentationMode::Interpolated { alpha }`.
Fractions must be finite and in `[0, 1]`; invalid requests reject before mutation.
Zero means the previous pose, one the current pose, for valid retained history.

Paused `step`, `render`, and ordinary `capture` render current authoritative state.
They never interpret the exact-step clock's zero remainder as a previous pose.
`present { alpha }` renders a controlled fraction without advancing clocks;
`capture { alpha }` also waits for presentation and captures that explicit fraction.
Adding `gpu: true` previews the same fraction through GPU presentation.
Previewing does not change the next ordinary capture's semantics.

Inspect `InterpolateTransform`, `PresentedTransform`, and `PresentationState`
through generated descriptors. Observations identify the last prepared extraction
mode, fraction, frame and tick, separate from current gameplay observations.
Extraction metadata is not GPU completion; a successful browser capture binds
an image to its render. Inspection refreshes authoritative globals without
overwriting presented poses.

## Cost and verification

History and presentation work remain in Rust. Storage and hierarchy traversal
are bounded by the transform-node limit (now 262,144) and reuse capacity.
Development observations separately report history entries and the two reserved
record buffers, retained presentation scratch, and presented-component payload.
Following [decision 011](adr-011-in-place-transforms.md), hierarchy source
validation also observes interpolation endpoints. The first source change in a
fixed epoch freezes the previous pose; later changes retain it while updating the
authoritative source. Presentation borrows that source as its current endpoint,
without retaining or copying a duplicate current pose. Tick begin is constant
work; tick end additionally checks a bounded,
coalesced list of validity transitions, normally empty. Unchanged records derive
their interval from the current epoch, so they never replay older motion.
Presentation does not repeat a history query. Structural changes reconcile
membership in one query pass and rebuild the entity index only when necessary.
Cached history and hierarchy slots validate full generational entity identity.
Records retain the previous pose and epoch/identity metadata; their bounded
storage is included in reported history bytes. The validity list
is included in scratch bytes. Explicit resets, marker/transform lifetimes, parent
changes and validity at the fixed endpoints preserve cancellation behavior.

Sparse fractional preparation visits blended
branches via existing child links. When opt-in histories reach at least 50% of
cached hierarchy nodes, preparation instead walks the existing parent-first
order and blends directly during composition. Dense preparation reads validated
source-to-history hints without a separate history scatter pass or touched-index
list, avoiding intermediate local-pose storage and repeated ancestor/work-stack
visits. This threshold is provisional;
matched browser measurements should guide changes to it.

Both paths use the same blend math and composition checks. Resets propagate
through unmarked descendants; invalid authoritative nodes and unsafe intermediate
matrices pass the shared validator and branch fallback. Dense preparation
initializes flags in its parent-first pass; sparse preparation resets touched
flags, and a dense-to-sparse transition clears the prior dense flags. Inactive
matrix slots are never consumed. Private matrix payloads carry a separate
validity flag, avoiding an aligned optional-matrix tag. Retained blended-local
and work/index vectors are included in scratch accounting. An ECS validation pass
still covers all presented components so direct replacements are repaired.
Every transformed entity carries the separate presented component. These counters
exclude hash maps, ECS table capacity/metadata, allocator overhead and process heaps.
No new per-entity host traffic, serialization, worker or GPU feature is required.

Native checks cover intermediate poses, multiple/zero-tick frames, hierarchy,
resets, invalid input, lazy-epoch expiry, repeated in-tick refreshes,
dense/sparse agreement, cached-index invalidation and
camera-edge visibility. The browser interpolation smoke
checks actual poses and captures through the agent interface. Benchmark `present`
mode repeatedly extracts alpha 0.5 without stepping; seed the same fixed ticks
when comparing it with current-state render mode. Static rendering alone does
not prove continuity of ordinary animation frames. Keep ordinary-frame coverage
and interpret costs using repeated browser measurements.
