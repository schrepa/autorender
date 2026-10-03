# Decision 049: Pipeline preparation

Status: in progress. The steps below land in order; this document records
the design and, under Measurements, what each step changed.

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

How other engines handle it:
- Unreal precaches the pipelines of each loaded asset (material × vertex
  factory × pass) and keeps the loading screen up until none remain.
- Godot 4.4 compiles a fallback set of ubershader pipelines when meshes load
  and specializes in the background, drawing with the fallback meanwhile.
- Bevy keys pipelines on fixed-function state and keeps texture presence as
  runtime material flags; Babylon.js and three.js expose a ready promise and
  count pipelines created per frame, which should be 0.

## Decision

### One preparation step, from a plan

`PipelinePlan` describes what a frame runs: its mesh passes with the
material variants and vertex classes their draw groups need, its screen
effects, shadow caster classes, lights, particles, decals, virtual geometry
and temporal antialiasing. It also holds what the content will run soon:
reflection-probe captures when the scene has probes, impostor bakes when
impostors are on.

`Renderer` builds missing pipelines for a plan in one place, inside an error
scope, before encoding. Encoding only looks pipelines up, and nothing is built
for features or modes a frame does not run.

### Warm-up behind the loading screen

`BrowserApp::prepare` extracts the first frame and prepares its whole plan.
The host keeps the canvas hidden meanwhile (Firefox paints a visible WebGPU
canvas by waiting on the GPU process), shows progress and starts frames once
the GPU queue has drained. `rendering.pipelines` counts resident pipelines and
`rendering.pipelines_created` those a frame built; steady frames build none.

### Specialize the hot path, share the rest

Main-view material variants stay specialized by override constants. Reading
the map flags from the material uniform instead cost 0.65 ms of
`opaque_meshes` (Firefox, integrated GPU), as Intel's guidance predicts.

Rarely drawn passes (reflection-probe captures, impostor bakes) use one uber
variant per pipeline-state key (vertex class, culling, alpha mask) that reads
map presence from the material uniform.

### Shared building blocks

Each WGSL module compiles once. Fullscreen and compute passes use common
builders, targets come from named sets, layouts are shared, a native test
checks every entry point name against the WGSL, and labels name the pass, the
entry point and the variant.

### Asynchronous where the browser compiles in parallel

A vendored wgpu (`vendor/wgpu`) carries asynchronous creation for warm-up.
Chrome compiles in parallel; Firefox, which compiles serially either way, is
unaffected.

## Measurements

Filled as the steps land.

## Alternatives and evidence

- **Ubershaders everywhere** (Godot's fallback model). Without background
  compiles in Firefox nothing would replace the fallback without a hitch, and
  the fallback alone costs 0.65 ms a frame on the hot path.
- **Skipping draws until a pipeline is ready** (Bevy, Unreal's default).
  The kept shadow map, reflection-probe faces and impostor bakes draw once;
  a skipped draw there stays wrong until the next refresh.
- **Creating at first draw** (the previous behavior). Every new pass or
  variant became a freeze in the middle of play.
