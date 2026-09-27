# Architecture decision 001: an observable browser engine slice

Status: adopted for the initial 3D milestone. The repository guide informs decisions; it does not mandate building every subsystem immediately.

This record describes the first milestone. [Decision 002](adr-002-assets-and-scenes.md) extends it with typed assets, generic mesh batches, authored scenes, action input, and explicit disposal; its scope and contracts take precedence where they differ.

[Decision 006](adr-006-hierarchy-and-prefabs.md) defines the current transform
propagation boundaries and prefab scene ownership. Initial scope statements below
are historical; the root README lists current capabilities.

[Decision 007](adr-007-visibility-and-benchmarks.md) adds conservative frustum
culling, source retention across visibility changes and repeatable browser measurements.

[Decision 008](adr-008-render-interpolation.md) adds opt-in local-pose history,
separate presentation transforms and explicit fractional agent previews.

[Decision 009](adr-009-collision-and-spatial-queries.md) adds opt-in collision,
fixed pre/post boundaries, authored collider data and agent spatial queries.

[Decision 011](adr-011-in-place-transforms.md) supersedes decision 010's immutable
transform storage. Ordinary mutable `Transform` queries and in-place derived
global writes use an engine-owned change-tick window. Change stamps are tested
before fetching payloads (now in contiguous blocks; see decision 013), and successful refreshes acknowledge derived
writes before advancing the ECS tick. Structural hooks retain membership
validation. `Parent` stays immutable, and `TransformEdits` remains available for
deferred replacements with lifecycle observers. Interpolation captures previous
endpoints during hierarchy validation and reads current poses from the shared
source cache, with constant-time normal tick begin/end bookkeeping. The decision
defines explicit change-detection opt-outs and the observer contract.

[Decision 012](adr-012-transform-validation-and-presentation.md) adds proven
common-case matrix admission, fused dense presentation, compact private matrix
storage, and a documented small-angle rotation interpolation policy. Public
matrix types, authoritative simulation, and agent inspection boundaries remain
the same.

[Decision 013](adr-013-canonical-poses-and-retained-rendering.md) makes
`GlobalTransforms` and `PresentedTransforms` the canonical read resources.
Derived pose values are no longer ECS components. Gameplay writes local poses;
agent pose projections and exact/fractional capture semantics retain their
existing schemas and boundaries. Borrowed retained render frames carry validated
change provenance while public owned snapshots remain supported. Stable GPU
pose/color slots and a compact visible-ID stream preserve uploads independently
of draw grouping. Dense interpolation uses bounded four-pose SIMD blocks before
parent-first composition.

[Decision 014](adr-014-gpu-presentation-and-culling.md) moves blending,
composition, validation and culling of ordinary interpolated browser frames to
compute passes with indirect draws. Core gathers per-node inputs only when ticks
or edits change them; frames that change only the fraction skip CPU presentation.
The CPU pass remains the exact reference for captures, previews and inspection,
and one asynchronous count read-back feeds statistics without GPU waits.
[Decision 015](adr-015-pipelined-fixed-ticks.md) splits each fixed tick into a
gameplay stage and a propagation stage run after consecutive frames are
submitted, so at twice the tick rate no frame runs a whole tick; GPU frames keep
presenting the previous interval until the tick is due.

[Decision 016](adr-016-deferred-leaf-globals.md) publishes changed leaves that
their parent's admission certificate proves valid without composing their world
matrices; reads compose them on demand with the same bits.

[Decision 017](adr-017-temporal-antialiasing.md) adds opt-in temporal
antialiasing of GPU-presented frames: jittered rasterization, per-pixel motion
from alternating pose buffers, and one resolve pass that writes the canvas and
the next history. Captures, previews and CPU-presented frames stay exact.

[Decision 018](adr-018-atmosphere-and-stylized-lighting.md) makes lighting
scene data: the `Atmosphere` resource (sun, sky colors, fog, grade) travels in
every snapshot, and the renderer adds sky-colored ambient and height fog,
filmic tone mapping and a sky pass behind geometry.

[Decision 019](adr-019-surface-materials-and-reflections.md) gives materials a
`Surface` (roughness, metalness, emission masked by texture alpha, ripples):
GGX sun highlights, reflections of the analytic sky, glow and liquid waves,
packed into the existing per-slot color record.
[Decision 020](adr-020-sun-shadows.md) adds sun shadows (`Shadows`): a compute
pass selects casters from every candidate into a near map redrawn per frame, a
far map of still casters redrawn only where dirty, and a far map of moving
ones. [Decision 021](adr-021-ambient-occlusion-and-bloom.md) adds ambient
occlusion and bloom (`AmbientOcclusion`, `Bloom`) through scene and glow
targets, applied in the temporal resolve or a composite pass. Captures,
previews and CPU-presented frames render all three alike.
[Decision 022](adr-022-scene-environments-and-agent-controls.md) gives the
look one JSON shape: a scene `environment`, its inspection projections and a
temporary agent control. Each effect resource is optional, and the renderer
creates an effect's GPU resources only while it runs.
[Decision 023](adr-023-local-lights.md) adds point and spot lights binned
into view clusters by a compute pass, transformed by the poses of the meshes
that carry them; [decision 024](adr-024-sun-shafts.md) marches the far
shadow maps through the fog at half resolution for sun shafts;
[decision 025](adr-025-screen-space-reflections.md) traces the depth buffer
at half resolution for screen-space reflections; and
[decision 026](adr-026-transparency-and-particles.md) blends transparent
materials and stateless GPU particles order-independently after the opaque
pass. Every pass shades captures, previews and GPU frames alike.
[Decision 027](adr-027-stylized-grading.md) grades every shaded color with a
hue-keeping filmic curve, contrast, split toning and vibrance.
[Decision 028](adr-028-shadow-cascades-and-receiver-planes.md) splits the near
shadow map into cascades and compares lookups against receiver planes.

## Ownership and compatibility

`engine` composes `engine_core` and `engine_render`. The renderer consumes core snapshots. Core imports neither rendering nor browser APIs. Platform glue is a module inside the façade until it needs a separate reusable interface. The demo imports the façade only. Host tools never become Wasm runtime dependencies.

The gameplay SDK is explicitly based on Bevy ECS and `bevy_app` 0.19.1. Components, queries, resources, entities, systems, and plugins are intentionally public Bevy types. We own the browser lifecycle, renderer, and scene contracts. We do not fork Bevy or promise compatibility with arbitrary Bevy plugins. `App::empty()` runs our named schedules, not Bevy's default `Main` schedule. Plugins register synchronously before the first frame; their finish/cleanup hooks are startup hooks. Duplicate plugin registration follows Bevy's unique-plugin checks. Runtime plugin removal is not supported.

Dependency versions and Rust are pinned, with Cargo/npm lockfiles. glam is patched with a vendored copy whose only change routes `f32` sine and cosine to the engine's branch-free `engine_math` ([engine math](math.md)). wgpu 30.0.1 handles WebGPU; generated wasm-bindgen 0.2.127 declarations define the browser boundary. The initial WebGPU profile requests no optional GPU features. The renderer and browser resources stay on one main thread; there is no shared-memory Wasm or cross-origin-isolation requirement.

## Frame contract

1. The host accumulates bounded held-key state and pointer deltas, and sends one owned input value per animation frame. There is no per-entity JavaScript traffic. Pointer motion is summed and clamped, not queued indefinitely. Discrete press/release event APIs are not implemented.
2. Rust computes elapsed time and runs `FixedSimulation` at 60 Hz, at most five ticks per frame. Commands apply at each schedule boundary. Extra whole ticks are discarded while the fractional remainder is kept. Counters report discarded time. Negative/nonfinite elapsed values are sanitized.
3. `Presentation` runs once after simulation. `FrameTime` caps its delta at 250 ms. Systems with dependencies must use explicit ordering. The demo camera uses this phase; cube motion uses fixed simulation.
4. Extraction copies transforms/colors into a reusable, owned snapshot, every visible instance unless `EngineConfig::max_render_instances` sets a budget (decision 029). Overflow is counted. Rendering cannot borrow the gameplay world. The available interpolation fraction is not yet used by this demo.
5. Rendering converts records into reusable staging storage, writes a camera uniform and one instance buffer, encodes a depth-tested instanced draw, and submits without GPU readback. WGSL and GPU record definitions live together. Buffer capacity grows geometrically within a hard limit.

Pause, focus loss, and hidden-page transitions release held inputs and reset the timing baseline. Visibility resumption does not simulate time spent suspended. A zero-sized renderer skips submission. Surface resizing recreates depth storage; surface loss/outdated status is reconfigured and device loss enters an explicit failed state.

## Costs and bounds

- Every frame performs O(N) fixed simulation work for each tick, extraction, and GPU record preparation. The sample uses one cube mesh and one material pipeline, with one indexed instanced draw.
- Extraction and staging retain capacity. Changing the workload may allocate and change ECS archetypes; it is an explicit control operation outside ordinary frame work.
- Runtime references use Bevy generational entities. They are not persistent scene identifiers.
- The host caps the pixel ratio at 1.5 and the drawing buffer at 2560 × 1440 while preserving aspect ratio. This controls fill cost independently of cube count.
- The demo caps its workload at 50,000 cubes; the engine sets no capacity of its own (decision 029): GPU buffer limits and memory bound content.
- Native checks validate contracts and WGSL. Native timing is not used to claim browser performance.

The HUD reports frame cadence, sampled CPU simulation/presentation and extraction time, draw calls, upload bytes, and sampled CPU render preparation/submission. It polls stats at 4 Hz and frees each generated binding object. The canvas diagnostics attributes report instance count, frame count, fixed steps, and cumulative discarded time. No GPU timestamps are claimed.

## Next milestones

1. Use the demo to establish browser/device baselines at fixed workload and viewport sizes, including warm-up, median, and tail frame times. Measure extraction and upload costs before adding parallelism.
2. Add a second game with authored geometry and textures. That requirement should drive typed asset handles, readiness/failure states, and a small versioned asset format.
3. Add cameras, visibility culling, simulation interpolation, and material batching where measured costs or gameplay needs justify them.
4. Introduce editable scene text and bounded development inspection before an editor or broad tool protocol. Preserve a clear difference between persisted authoring changes and temporary runtime edits.
5. Evaluate workers, asset compilation, render passes/graph, recovery, and mobile quality tiers as concrete workloads require them.

## Primary references

- [Bevy ECS 0.19.1](https://docs.rs/bevy_ecs/0.19.1/bevy_ecs/) and [Bevy App](https://docs.rs/bevy_app/0.19.1/bevy_app/): public gameplay foundation.
- [wgpu 30.0.1](https://docs.rs/wgpu/30.0.1/wgpu/): browser GPU ownership and rendering API.
- [Cargo workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html): shared dependencies, resolver, and lints.
- [wasm-bindgen](https://wasm-bindgen.github.io/wasm-bindgen/): generated browser bindings.
