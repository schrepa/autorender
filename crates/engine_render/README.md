# engine_render

Owns the browser's WebGPU surface, device, depth attachment, shaders,
pipelines, GPU meshes, textures and materials, uniforms and slot storage. The
`engine` composition layer owns a `Renderer`, which draws the core-owned
`RenderFrame` each frame borrows from `engine_core::Engine`. This crate never
reads or borrows the ECS world, and private GPU types do not enter the
ordinary gameplay API.

`Renderer::new(canvas).await` creates a WebGPU renderer on the browser main
thread. `resize(width, height)` accepts physical pixels; zero dimensions suspend
submission, and oversized dimensions fit the device limit while preserving
aspect ratio. `render_frame(frame)` draws a frame exactly: never jittered or
blended with earlier frames. `render_display_frame(frame)` draws an ordinary
display frame, which `set_temporal_antialiasing(true)` temporally antialiases
when it is GPU-presented ([decision 017](../../docs/adr-017-temporal-antialiasing.md)).
Both return `RenderStats` (one definition, serialized under the same names for
hosts and agents) for the submitted work and current GPU asset residency;
`is_texture_resident(handle)` observes uploaded storage without waiting for the
GPU. The browser API exists only on `wasm32`; native tests exercise batch
planning, budgets, binding tables against every shader, shader validation,
upload layouts, normal transforms and dimension bounds.

## Frames and slots

A frame's instance records sit at stable core slots. Its visible indices list
the slots it submits; inactive slots are never submitted or validated as draws.
Frames certify membership and sources with a stream identity and structure and
source revisions. The batch plan groups visible slots by mesh, material and
opacity class with a counting scatter that keeps slot order within each group,
and opaque groups draw sorted by pipeline variant, material and arena buffers
(decision 046), so passes switch each once per run. An unchanged
structure revision reuses the previous order and ranges; each slot keeps its
draw-group ID for its source revision, so visibility changes scatter those IDs
without resolving meshes and materials again. Sources and budgets are
validated whenever the structure changes, the camera's finiteness on every
frame, and an invalid frame returns an error before any upload or cache change.

Slots' poses live in a 48-byte storage pose buffer (three affine rows; the
shader restores the constant final row, so shear, reflection and every affine
coefficient survive without quantization) and their colors in a 16-byte buffer
(color and opacity as four 16-bit fractions, the packed `Surface` and the
skinned palette offset). A four-byte slot-ID vertex stream selects visible
slots in draw order, so visibility changes rewrite only IDs. Storage grows in
powers of two up to what one storage binding holds and never shrinks; that
binding bounds the slots a frame draws.

Uploads compare core-issued per-slot pose and color stamps rather than bytes.
Core writes a slot's `RenderFrame::pose_rows` entry in the 48-byte layout when
its pose stamp advances, and changed ranges upload straight from those rows: the
renderer keeps no CPU pose copy and stages only colors. Unchanged channel
revisions skip even the stamp scan. Each stream plans at most 32 ranges, merging
clean gaps within a 16 KiB allowance per queue call (342 poses or 1024 colors)
and falling back to a full write for fragmented plans or ones covering 75% of
the buffer. Revisions stay usable across skipped frames; a new stream identity,
replaced buffer or uncommitted preparation forces a complete refresh, and reuse
is committed only after every planned write reached the queue. The camera
uniform (240 bytes) and lighting uniform (176 bytes) are rewritten only when
their bytes change.

Pose uploads of at least 256 KiB copy into a persistently mapped staging buffer
(`MAP_WRITE | COPY_SRC`) and GPU copies submitted before the frame's draws:
Firefox backs every `writeBuffer` of 1 KiB or more with new shared memory, so a
multi-megabyte write would fault in fresh pages every frame. Bytes go from Wasm
memory straight into the mapping through a JS view. At most six staging buffers
exist, each sized to the largest upload in 256 KiB steps; maps resolve
asynchronously, the renderer never waits, and it falls back to `writeBuffer`
when none is mapped.

Frames whose `RenderFrame::gpu()` is present use GPU presentation
([decision 014](../../docs/adr-014-gpu-presentation-and-culling.md)). Node inputs
upload only records whose core stamps are newer than the last uploaded gather
(at most 64 coalesced ranges, a full write beyond 75% density, staged from
256 KiB), and candidate and draw-group records only when the frame's structure,
source revision or input identity changes. `presentation.wgsl` composes active
parents level by level, blends, composes, validates and culls every candidate,
writes its pose rows into the slot pose buffer, and compacts visible slot IDs
per draw group; each group draws with `draw_indexed_indirect` from its region of
the ID buffer (named as the first instance where the adapter supports
`indirect-first-instance`). The CPU writes one 320-byte uniform and records a
few dispatches per frame; visible counts are read back a few frames late
(`RenderStats::gpu_visible`). The shader's culling and admission constants are
exact hex literals tested against core's values. `prestage_gpu_current` copies
current poses into a mapped staging buffer between frames when their gather
stamped most of them.

## Meshes, materials and textures

Meshes are validated indexed triangle lists of 24-byte vertices: position,
octahedral normal and texture coordinates ([decision 032](../../docs/adr-032-compact-vertices.md)).
Meshes with vertex colors or second texture coordinates add a second stream
read through `_extended` entry points (decision 034), and meshes with joints a
skin stream (decision 036). Every mesh's indices and vertex streams live in
shared arena buffers per stream class (`mesh_arena.rs`, decision 046), so passes
bind them once and draws select ranges.

Materials are glTF metallic-roughness materials
([decision 031](../../docs/adr-031-gltf-materials.md)): base color,
metallic-roughness, normal, occlusion and emissive maps with their own samplers
(16x anisotropic when trilinear) and UV transforms. Materials without maps share
the basic pipeline variant and batch across materials; materials with maps draw
grouped by binding, each with its variant. Texture files are RGBA8 sRGB, whose
base level uploads and whose mip chain the GPU renders when they become resident
(`mipmaps.rs`), or KTX2 levels the host transcoded to the adapter's BC7, ASTC
or ETC2 format ([decision 037](../../docs/adr-037-compressed-textures.md)),
uploaded as they are. Untextured slots use a renderer-owned white texel.

An asset uploads only when a submitted slot references it. A frame's sources
include those of culled valid objects, so every valid render boundary, including
suspended and occluded frames, keeps their GPU resources resident: a culled
object never triggers its first upload, and an object moving out of view and
back reuses its resources. Resources retire when their source leaves the
frame's sources. Only the GPU's limits bound content
([decision 029](../../docs/adr-029-budgets-and-device-limits.md)); texture
dimensions cannot exceed the device limit (8,192 in the CPU asset contract).

## Shading and effects

Slots are lit by the frame's `Atmosphere`
([decision 018](../../docs/adr-018-atmosphere-and-stylized-lighting.md)) and
graded ([decision 027](../../docs/adr-027-stylized-grading.md)), with a sky pass
over uncovered pixels. Optional features run only while their settings enable
them, and create their GPU resources and pipelines then: sun shadows with near
cascades and receiver planes (decisions 020 and 028), ambient occlusion and
bloom (021), local lights binned into 16x9x24 view clusters by a compute pass
reading slot lights' poses from the frame's pose buffer (023), sun shafts (024),
screen-space and environment reflections with probes (025, 043), weighted
blended transparency and stateless particles (026), glass refraction (038),
decals (039), occlusion and sub-pixel culling (040, 041), levels of detail and
impostors (047, 048), and virtual geometry with its visibility buffer (044, 045).
Inputs use right-handed world coordinates, reversed WebGPU depth (decision 042)
and linear RGB colors; normals follow the cofactors of each slot's matrix.

Temporal antialiasing (`temporal.rs`, `temporal.wgsl`) renders GPU-presented
display frames into a scene target and an `Rg16Float` motion target with a
Halton sub-pixel jitter, culling with the same jittered projection. Culling
alternates between the slot pose buffer and a second one, so the previous
frame's poses stay readable; visible IDs of moving slots carry `MOVED_SLOT` and
only those read their previous pose. One full-screen pass blends with the
reprojected history and writes the canvas and the next history.

Each frame declares its passes in a render graph (decision 050) and pipelines
are prepared as the frame first needs them (decision 049): every render and
compute pipeline is counted, and `pipelines_compiling()` stays true until the
GPU completes the last frame that created some, so hosts reveal the canvas once
compiles end.

## Device and failures

`new` requests block-compressed texture formats, `indirect-first-instance` and
(in profiling builds) timestamp queries where the adapter offers them, the
adapter's buffer and storage binding sizes, at most 8,192-texel 2D textures and
at least eight storage buffers per shader stage. WebGPU on HTTPS or localhost is
required; WebGL is not a fallback. An outdated surface is reconfigured at the
next render boundary and a lost surface is recreated. Device loss or uncaptured
GPU errors place the renderer in an explicit failed state; the host displays the
error and asks for a reload.

## Statistics

`upload_bytes` reports actual camera and slot writes plus newly initialized mesh
and texture bytes; it can be zero on a rendered frame and does not estimate total
GPU traffic. `mesh_upload_bytes` and `texture_upload_bytes` report first uploads
this frame; the `resident_*` fields describe caches after retirement and
uploads. A skipped frame reports zero work counters and current residency. These
are workload counters, not performance claims. Profiling counters (pose, color
and ID write bytes and ranges, staging use, GPU presentation work, memory
capacities) are listed in [`docs/profiling.md`](../../docs/profiling.md).

Dependencies: core frame contracts, `glam`, `bytemuck`, and target-specific
`wgpu`, `web-sys` and `wasm-bindgen`; Naga is a shader-validation test
dependency. Wgpu usage follows the pinned [30.0.1 API](https://docs.rs/wgpu/30.0.1/wgpu/).
