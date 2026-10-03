# Decision 049: Pipeline preparation

Status: accepted, apart from asynchronous creation (not yet). Measurements
records what each step changed.

## Context

Every feature since decision 001 created its own GPU pipelines where it was
first needed. Twenty-six creation sites in fifteen modules ended up building
pipelines either eagerly at startup (GPU presentation, virtual geometry,
environment maps, mipmaps, the base mesh pipelines) or lazily in the middle
of a frame (material variants, lights, screen effects, particles, decals,
and reflection-probe variants during encoding).

Measured on the Stress Lab Metro (Intel Xe-LPG):
- The page builds 13 shader modules and 103 pipelines. 23 are never bound in
  Metro, about 25 are used only occasionally, and the 12 reflection-probe
  pipelines are created about 3 s into the session.
- Compiled one after another, the captured descriptors take 12.8 s in
  Firefox and 6.0 s in Chrome (`artifacts/pipeline-replay`, cold, with a
  presenting frame loop).
- In Firefox the first frame appears after 9-13 s and the page then freezes
  for up to 20 s while the GPU process compiles. Chrome is responsive after
  about 15 s on a first visit and 2.6 s on later visits, from its disk cache.

What browsers allow:
- WebGPU takes WGSL only; there is no way to ship compiled shaders
  (gpuweb#14). Every distinct set of `override` values is a full compile, in
  both browsers.
- Firefox compiles every pipeline, including `createRenderPipelineAsync`
  ones, serially on its WebGPU thread, so frames wait behind compiles; it
  keeps no cache across sessions. Six main-view variants: 908 ms blocking,
  726 ms asynchronously, with frames stopped in both cases.
- Chrome compiles asynchronous pipelines on worker threads: the same six
  variants resolved in 180 ms with no frame over 9 ms, against an 875 ms
  stall when created blocking. Its disk cache serves later visits.
- wgpu 30 creates pipelines only blocking; PR gfx-rs/wgpu#10438 adds
  asynchronous creation.
- Pipeline layouts cost as much as shaders in Firefox. It validates indirect
  calls, so wgpu-core marks every pipeline layout `INDIRECT_BUILTIN_UPDATE`
  and wgpu-hal's Direct3D 12 backend creates three indirect command
  signatures with root constants for each. The integrated Intel GPU's driver
  compiles something for each new root signature: about 150 ms per distinct
  layout in Firefox (7 ms for one equal to an earlier one), 168-602 ms in
  native wgpu 30 with indirect validation and none without it. A native
  process keeps the driver's results in its disk cache; Firefox's sandboxed
  GPU process never does. Chrome creates the same layouts in 2 ms in all.
  Root signatures depend only on each group's sequence of uniform, read-only
  and read-write descriptors, so layouts whose bindings match that way
  share one; `layout: None` derives a new layout for every pipeline.

How other engines handle it:
- Unreal precaches the pipelines of each loaded asset (material × vertex
  factory × pass) and keeps the loading screen up until none remain.
- Godot 4.4 compiles a fallback set of ubershader pipelines when meshes load
  and specializes in the background, drawing with the fallback meanwhile.
- Bevy keys pipelines on fixed-function state and keeps texture presence as
  runtime material flags; Babylon.js and three.js expose a ready promise and
  count pipelines created per frame, which should be 0.

## Decision

### Pipelines prepared with the frame that first runs them

Every render and compute pipeline goes through `pipelines::render` and
`pipelines::compute`, which count creations. A frame's setup creates what
its passes run, before they record: `ensure_variants` the material variants
and vertex classes of its draw groups and the sky of its targets, and each
pass module its own pipelines (screen effects, shadow caster classes,
lights, particles, decals, virtual geometry, temporal antialiasing) when the
frame first uses it. What the content needs soon is prepared with the first
frame that has it: reflection-probe captures with the first frame that has
probes, impostor bakes when a draw group first draws as an impostor. The
passes only look pipelines up. Nothing is built for features or modes no
frame has run, layouts included: mesh pipeline layouts are created with the
first variant needing each, public snapshots' shadow casters with the first
snapshot frame.

### Warm-up behind the loading screen

Ordinary sessions hide the canvas from the start (Firefox paints a visible
WebGPU canvas by waiting on its GPU process). The host renders the starting
state with zero elapsed time until frames stop creating pipelines and the
GPU has completed the last frame that created some (status `preparing`),
then reveals the canvas and starts play. `rendering.pipelines` counts the
pipelines created and `rendering.pipelines_created` those a frame created;
steady frames create none.

### Specialize the hot path, share the rest

Main-view material variants stay specialized by override constants. Reading
the map flags from the material uniform instead cost 0.65 ms of
`opaque_meshes` (Firefox, integrated GPU), as Intel's guidance predicts.

Rarely drawn passes (reflection-probe captures, impostor bakes) use one uber
variant per pipeline-state key (vertex class, culling, alpha mask) that reads
map presence from the material uniform.

### Shared building blocks

Fullscreen passes build through `pipelines::fullscreen` and compute passes
through `pipelines::compute_entry`; mesh pipelines are labeled with their
entry points and variant bits.

Pipelines declare their layouts; none derive one (`layout: None`). Passes of
a family share one layout that lists every binding any of them reads, built
from `pipelines::Slot`s, and each pass's bind group fills the bindings it
leaves alone with `pipelines::Fillers`: 1x1 textures, small buffers, and a
range of their own for written storage, since written bindings of a dispatch
or draw may not overlap. Every pass of post.wgsl shares one layout (16
bindings, within WebGPU's default 16 sampled textures per stage); GPU
presentation's twelve passes share five and virtual geometry's ten share
four, each within the default eight storage buffers per stage, and passes
binding the same buffers (GPU presentation's four compaction passes per
compaction, virtual geometry's dispatch-argument passes) share one group.

### Not yet: asynchronous creation where the browser compiles in parallel

Chrome compiles `createRenderPipelineAsync` pipelines on worker threads, so
warm-up there could create the frame's pipelines asynchronously and wait
for all of them. wgpu 30 only creates them blocking (PR gfx-rs/wgpu#10438
adds asynchronous creation); this needs a patched wgpu and a setup that can
wait for pipelines. Firefox compiles serially either way.

## Measurements

Stress Lab Metro, Intel Xe-LPG:

| Step | Pipelines (modules) | Created during play |
|---|---|---|
| Before | 103 (13) | 12, about 3 s in |
| Base pipelines as variants, virtual geometry on first use | 83 (12) | 12 |
| Uber variants for probe captures and impostor bakes | 74 (12) | 6 |
| One preparation step, probes with the first frame, loading gate | 74 (12) | 0 |
| Shared layouts: post 10 to 1, presentation 12 to 5 | 74 (12) | 0 |
| Layouts and snapshot shadow casters by need; every layout declared | 72 (12) | 0 |

- Never bound in 25 s of play: 23 before, 5 after the first step (three
  shadow pipelines of other paths and two bloom passes the reflection levels
  replace).
- Startup, cold profiles, until the canvas shows a playable frame:
  - Firefox: first frame after 9-13 s, then freezes of 16-20 s (playable
    after 12-35 s) before; playable after 14.0-14.5 s with the loading gate,
    with no frame over 120 ms in the 15 s that follow.
  - Compaction in workgroups of 64 cut Firefox's compiles of the
    presentation passes from 3.29 to 2.37 s; time to playable stayed within
    noise. A profile showed why: the GPU process compiles for 10-14 s, 92% of
    it in the driver's compiler (IGC, 74%) and DXC (18%), while the page's
    thread waits. Hiding the canvas from the start freed the page's thread
    (it no longer waits about 1 s to paint) and changed nothing else.
  - Shared layouts: Firefox playable after 11.0-12.3 s against 13.3-14.5 s
    (three alternating cold runs each, paired differences -2.2 to -2.3 s).
    Cold replays of the captured layouts and pipelines: 9.1-9.4 s against
    10.7-10.9 s, the presentation passes 0.50-0.57 s against 2.38-2.40 s
    with their five layouts 0.81 s, and post 0.27 s against 1.36 s.
  - Mesh pipeline layouts created by need (three of four were unused), the
    public snapshots' shadow casters with the first snapshot frame, and
    every layout declared (the depth pyramid's and environment's now share
    shapes on purpose; virtual geometry's ten passes share four layouts
    instead of six derived ones): Firefox playable after 10.3-11.1 s against
    11.4-12.0 s (three alternating cold runs each, paired differences -0.3
    to -1.7 s); replays 8.6-9.1 s against 9.4-10.0 s. 22 distinct layouts
    remain, about 3.7 s of the replays in Firefox.
  - Chrome: playable after about 15 s before; 6.8 s with the loading gate,
    with no stall after.
  - With every step (7b449a7): Firefox playable after 10.5-10.9 s against
    13.2-14.2 s for the loading gate with workgroups of 64 (three alternating
    cold runs each), no frame over 100 ms in the 10 s after; Chrome after
    5.8-6.2 s. Firefox's GPU process still compiles serially for about 8 s:
    21 distinct layouts (about 3.2 s) and 72 pipelines.
- Every step kept captures pixel-identical (overview and close view, CPU and
  GPU paths, with and without temporal antialiasing, with and without
  reflection probes); GPU-presentation, TAA, LOD, virtual-geometry and web
  smokes pass.

## Alternatives and evidence

- **Ubershaders everywhere** (Godot's fallback model). Without background
  compiles in Firefox nothing would replace the fallback without a hitch, and
  the fallback alone costs 0.65 ms a frame on the hot path.
- **Skipping draws until a pipeline is ready** (Bevy, Unreal's default).
  The kept shadow map, reflection-probe faces and impostor bakes draw once;
  a skipped draw there stays wrong until the next refresh.
- **Creating at first draw** (the previous behavior). Every new pass or
  variant became a freeze in the middle of play.
