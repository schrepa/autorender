# engine_render

Owns the browser's WebGPU surface, device, depth attachment, shader, pipeline,
mesh and texture caches, uniform, and instance buffers. The `engine` composition layer owns a
`Renderer`; the normal browser path borrows a core-owned `RenderFrame`, while
custom clients can still provide an owned `engine_core::RenderSnapshot`.
This crate never reads or borrows the ECS world. Private GPU types do not enter
the ordinary gameplay API.

`Renderer::new(canvas).await` creates a WebGPU renderer on the browser main
thread. `resize(width, height)` accepts physical pixels; zero dimensions suspend
submission, and oversized dimensions fit the device limit while preserving
aspect ratio. `render(&snapshot)` returns `RenderStats` for submitted work and
current GPU asset residency. `render_frame(frame)` accepts the private core-owned
revision certificate and uses compact affine uploads; it is always exact.
`render_display_frame(frame)` renders an ordinary display frame, which
`set_temporal_antialiasing(true)` temporally antialiases when it is GPU-presented
([decision 017](../../docs/adr-017-temporal-antialiasing.md)). `is_mesh_resident(handle)` and `is_texture_resident(handle)`
observe uploaded storage without waiting for GPU completion. The browser API exists only on `wasm32`; native
tests exercise batch planning, budgets, asset retirement, shader validation,
upload layouts, normal transforms, and dimension bounds.

Each snapshot owns unique `RenderMesh` and `RenderTexture` sources through `Arc`, keyed by
library-specific generational handles. Meshes are immutable, validated indexed
triangle lists: positions, normals and UV coordinates, with 32-bit indices. A handle must always
refer to the same immutable asset. There is no implicit renderer cube; the
convenience cube uses the same asset path as user meshes.

The renderer groups instance indices by mesh and optional texture handle and
retains the ordering and draw ranges. Engine frames certify membership and
sources with a stream identity and structural revision; unchanged revisions skip
source rebuilding and instance-key comparisons. A separate source revision
certifies mesh/texture keys independently of visibility. Each stable slot caches
its draw bucket; visibility changes count and scatter those cached bucket IDs,
resolving only slots appearing for the first time in that source revision.
Their visible indices select
stable source slots, so inactive records are never submitted. Public snapshots
retain ordering only while their exact input-order keys match.
Pose/color edits preserve batching; changed membership or keys rebuild it.
Counting scatter preserves source order within each group, including coherent
static/moving runs, while draw groups remain sorted by mesh/texture handle.
Source lists and budgets are validated on every public snapshot and each new
engine structural revision. Camera finiteness is checked on every call. Source
indices are refreshed after source-list edits. Public transform/color records
use one reusable staging vector, with one contiguous
draw range per mesh/texture pair. Material colors are per-instance data, so color variations share a
draw. Instances are opaque, with depth testing and back-face culling, lit by
the snapshot's `Atmosphere` ([decision 018](../../docs/adr-018-atmosphere-and-stylized-lighting.md)):
sky-colored ambient light, a wrapped sun, a rim light on walls, height fog in
the color of the sky haze, filmic tone mapping and saturation, and a sky pass
over uncovered pixels. The atmosphere uploads as a 112-byte uniform only when
it changes. Each slot's color record packs its `Surface`
([decision 019](../../docs/adr-019-surface-materials-and-reflections.md)): GGX
sun highlights, sky reflections by roughness and metalness, emission masked by
texture alpha, and rippling normals; base colors sample 16x anisotropically.
Sun shadows ([decision 020](../../docs/adr-020-sun-shadows.md)) come from three
16-bit maps whose casters a compute pass selects from every candidate, and
ambient occlusion and bloom ([decision 021](../../docs/adr-021-ambient-occlusion-and-bloom.md))
run on the scene and glow targets, applied by the temporal resolve or a
composite pass. Inputs use right-handed world coordinates,
a WebGPU 0..1 depth projection, linear RGB colors, and finite, nonsingular model
transforms with positive determinant. Shader cofactors preserve normals under
nonuniform positive scale. These transform requirements are client obligations;
the renderer validates source references, budgets, and camera finiteness rather
than adding another per-instance transform-validation pass. Core performs
conservative frustum culling before extraction. Transparency, shadows of
other lights and render extension hooks remain future work.

Engine frames guarantee affine transforms and use three four-float rows in a
48-byte storage pose buffer plus a separate 16-byte storage color buffer. Both
arrays retain the core's stable slot positions. A four-byte slot-ID vertex stream
selects visible instances in grouped draw order. Visibility changes therefore
rebuild only IDs, without moving poses or retransmitting unchanged colors.
This preserves shear,
reflection and all affine coefficients without quantization; the shader restores
the constant final row. Ordinary motion uploads poses while unchanged colors
stay resident. The public snapshot path keeps its original full 80-byte matrix/
color format, including arbitrary fourth matrix rows. Both vertex entrypoints
share the same position, normal and lighting implementation.

Retained upload state compares core-issued per-slot pose/color stamps rather
than matrix/color bytes. Changed poses are not packed here: core writes each
slot's `RenderFrame::pose_rows` entry in this exact 48-byte layout when its
pose stamp advances, during the extraction pass that already reads the matrix,
and the selected ranges upload directly from those rows. The renderer keeps no
CPU pose copy; only colors are staged. Unchanged global channel revisions skip even the stamp
scan. Changed channels scan slots in ascending order, including hidden slots.
Revisions remain usable across skipped frames; a changed stream identity,
GPU buffer replacement or uncommitted preparation forces complete refresh.
New slots initialize only their appended records while buffer capacity fits.
Calling the public snapshot renderer invalidates retained upload trust. Switching
layouts drops the previous CPU staging representation instead of retaining two
full instance payload populations.

Pose uploads of at least 256 KiB copy into a persistently mapped staging buffer
(`MAP_WRITE | COPY_SRC`) and record GPU copies submitted before the frame's
draws. Firefox backs every `writeBuffer` of 1 KiB or more with newly created
shared memory, so a multi-megabyte write faults in fresh pages every frame;
a mapped buffer keeps one mapping for its lifetime. Bytes go from Wasm memory
straight into the browser's mapping through a JS view, without wgpu's Rust-side
copy of the whole mapping. At most three staging buffers exist, each sized to
the largest upload in 256 KiB steps. Maps resolve asynchronously after the GPU
releases a buffer; the renderer never waits, and falls back to `writeBuffer`
when none is mapped. Replacing the instance buffer drops the ring. Smaller
uploads and colors keep `writeBuffer`, which is cheap on every browser.

Frames whose `RenderFrame::gpu()` is present use GPU presentation
([decision 014](../../docs/adr-014-gpu-presentation-and-culling.md)). Node inputs
upload only the records whose core stamps are newer than the last uploaded gather
(at most 64 coalesced ranges; a full write beyond 75% density; staged at 256 KiB
or more), planned from the gather's listed nodes through a node bitmap when it
listed them (`GpuPresentation::stamped_nodes`) instead of scanning every stamp,
and candidate/draw-group records only when the frame's structure,
source revision or input identity changes. `presentation.wgsl` then composes
active parents level by level, blends, composes, validates and culls every
candidate in slot order (its records stream in memory order), writes its 48-byte
pose rows into the same slot-indexed pose buffer and its visibility flag at its
draw-group position, and compacts visible slot IDs per draw group with a
two-level scan. Each group
draws with `draw_indexed_indirect` from its own region of the ID buffer, so no
`indirect-first-instance` feature is needed. Per frame, the CPU writes one
208-byte uniform and records a few dispatches. The instance budget is applied in
draw-group order. A four-byte visible total is copied into one of three readback
buffers and mapped asynchronously; `RenderStats::gpu_visible` reports the latest
completed value, and `instances` limits it by the budget. Switching back to CPU
frames re-uploads poses and the slot-ID stream. Buffers grow to powers of two
and never shrink; all node, candidate and group storage appears in
`memory.gpu_presentation_bytes`. The shader's culling and admission constants are
exact hex literals tested against core's values.

Base color textures are opaque, tightly packed RGBA8 sRGB with upper-left pixel
and UV origins. The renderer uses `Rgba8UnormSrgb`, decoding samples to linear RGB
before multiplying the material's linear color and lighting. A shared sampler
uses clamp-to-edge and trilinear filtering. Each texture uploads its base level
only; the frame that makes it resident renders the rest of its mip chain on the
GPU (`mipmaps.rs`, one small pass per level, each averaging the level above in
linear space through the sRGB format), submitted before the frame's draws. The
chain adds at most a third of the base bytes in GPU memory
(`memory.gpu_texture_mip_bytes`); `render.texture_mip_levels` counts levels
generated in a frame. Minified textures then read small, cached levels: in
Firefox on the Intel iGPU the opaque pass stayed within run-to-run variation
(1.92–2.01 ms without mipmaps, 1.93–1.99 ms with them at 90,000 textured
objects; 5.77 against 5.51–6.05 ms at 250,000). Each resident texture owns one
GPU texture and cached bind group; untextured instances use a renderer-owned
white texel. There is no normal map, anisotropy or per-material sampler
specialization in this milestone.

An asset uploads only when an emitted instance references its handle and that
handle is absent from its GPU cache. Snapshot mesh/texture sources include both
emitted and culled valid objects. Every valid render boundary retains existing
cache entries for all supplied sources, including suspended, timed-out, and
occluded frames. A culled object never triggers its first upload; an object that
moves out of view and back reuses its existing GPU resources. Cache entries
retire when their source disappears from the snapshot. Wgpu retains resources
needed by previously submitted GPU work. Dropping a scene therefore removes its
unused GPU assets on the next render boundary; a subsequently reused asset is
uploaded again. Invalid snapshots return an actionable error before GPU cache
changes. Snapshots cannot exceed 262,144 instances, 128 source meshes, or 32 MiB
of total vertex/index data, independently of 64 source textures and 32 MiB of
RGBA8 texture data. Texture dimensions cannot exceed the device limit (the CPU
asset contract further caps each dimension at 2048). Missing or duplicate source handles also fail.
Source CPU ownership remains with the library/snapshot; renderer cache entries
own GPU resources only. The renderer's permanent four-byte white texel is outside
asset budgets and residency/upload counters.

Changed batching keys take O(N * (log(M + 1) + log(T + 1)) + M * (T + 1)
+ M log M + T log T) work for N instances, M meshes and T textures.
Matching public keys take O(N + M log M + T log T +
B log(M + T)) for B retained batches. A matching core structural revision skips
the key scan and batch/source rebuild. There are no per-entity heap allocations.
Visibility-only regrouping takes O(N + M * (T + 1) + M log M + T log T) for
already mapped slots, without resolving mesh/texture handles again. Profiling
reports `render.batch_key_resolutions` and `render.batch_slot_mapping_reused`
to distinguish that work from a source change.
Camera bytes are compared with the last queued 144-byte uniform (temporal
frames change it every frame: jitter and previous view-projection). For public snapshots, each instance's
matrix/color bytes are compared with the existing 80-byte staging record;
unchanged records are not repacked. Changed records produce at most 32 upload
ranges. Clean gaps of at most 205 records are included to avoid small queue
writes. A full write replaces fragmented plans, plans covering at least 75% of
the buffer, or plans whose byte savings do not exceed a 16 KiB allowance per
additional queue call. These bounded heuristics require browser measurement;
interleaved motion may still upload the whole buffer. Changed counts, fresh
buffers, and an earlier uncommitted preparation force a full write. Reuse is
committed only after every planned write reaches the queue. Unchanged frames
still draw normally with zero camera/instance
uploads. Newly resident meshes also
initialize their own vertex/index buffers; newly resident textures use queue
writes with tightly packed rows (no 256-byte buffer-copy row padding). Vertex attributes, column-major
uniform bytes, and `meshes.wgsl` are co-located; byte-cast eligibility is not used
as evidence of WGSL layout. Each vertex is 32 bytes and each index is 4 bytes.

Instance staging and the GPU instance buffer grow geometrically within the
262,144 instance limit (21 MB each for public snapshots). The retained path uses
at most 1.6 MB of color staging (pose rows are core-owned) and 6.4 MB of GPU
instance storage, plus 0.4 MB each of CPU/GPU slot IDs, 1.6 MB of per-slot
revision stamps and at most three mapped pose staging buffers, each the largest
pose upload rounded up to 256 KiB (under 5 MB; reported as
`memory.gpu_pose_staging_bytes`). It plans
at most 32 pose ranges and 32 color ranges independently, plus one ID write,
using the same 16 KiB call allowance and 75%
density threshold. Clean-gap limits follow each stream's stride: 342 pose or
1024 color records. Dense motion can therefore upload every pose without
uploading unchanged colors. Fixed range scratch is 512 bytes for public
snapshots plus 1024 bytes for retained frames on Wasm; these are included in
upload-cache memory counters. Planning retains at most
262,144 native-size indices and exact mesh/texture key pairs, plus 128 mesh
source indices and retained handles, 64 texture source
indices and retained handles, and at most
128 * 65 draw ranges. Stable grouping additionally retains at most 262,144 u16
visible bucket IDs, 262,144 u16 cached slot buckets, and 8,320 native-size cursors
(433,280 bytes on Wasm). A reusable vertex conversion
vector retains capacity for the largest source encountered, bounded by the mesh
byte budget. The at-most-128-entry GPU cache allocates on new mesh residency,
not on every draw. The texture cache is at most 64 entries. GPU command objects and presentation textures are created
per frame. GPU-presented frames read back one four-byte count asynchronously;
nothing waits on the GPU.

`upload_bytes` reports actual camera/instance writes plus newly initialized
mesh/texture bytes. It can be zero on a rendered frame and does not estimate
total GPU traffic. `mesh_upload_bytes` reports only first
uploads this frame, and `texture_upload_bytes` reports newly uploaded texels.
`resident_meshes`, `resident_mesh_bytes`, `resident_textures`, and
`resident_texture_bytes` describe caches after retirement and any uploads. A skipped frame reports zero work
counters and current residency. These are workload counters, not performance
claims; browser workloads must establish timing baselines.

Profiling records actual instance write bytes/range counts, full writes, and
fragmentation/density fallbacks. Retained frames additionally report pose/color
write bytes and changed records separately; `render.poses_repacked` counts pose
slots selected for upload (their rows come from core, not repacking).
`render.pose_staged` is 1 when a large pose upload used a mapped staging buffer
and `render.pose_staging_unavailable` is 1 when none was mapped and it fell back
to `writeBuffer`. GPU-presented frames report `render.gpu_presentation`,
`render.gpu_candidates`, node and group upload bytes, whether node inputs needed
a full write or used staging, whether current poses came from
`prestage_gpu_current` (`render.gpu_current_prestaged`), parent depth levels and
`render.gpu_visible_readback`; their `render.instances` is the budgeted read-back
count. `prestage_gpu_current(current)` copies current poses into a mapped
staging buffer between frames (`prestage_current` scope) when their gather
stamped at least three quarters of them (`GpuCurrent::stamped`); the next GPU
frame whose inputs have that version and upload every current pose copies them
on the GPU instead of staging them again. Gathers stamping fewer poses upload
only those records in the frame. A view with `GpuPresentation::hold` (inputs
gathered ahead of their due frame) applies nothing: the pass presents the inputs
applied last at the view's fraction (`render.gpu_held`), and a later view without
`hold` applies the gather. `render.instance_full_upload`
means at least one instance stream needed a full write. `memory.render_upload_range_bytes` is a subset
of `memory.render_upload_cache_bytes`; `memory.batch_group_scratch_bytes` is a
subset of `memory.batch_plan_capacity_bytes`. Group rebuild timing is named
`batch_instance_group`, replacing the previous `batch_instance_sort` scope.

GPU objects stay in their creating execution context. `new` requests no optional
GPU features and uses downlevel limits, with a 2D texture dimension request
bounded by the adapter and 8192 and eight storage buffers per shader stage (the
WebGPU default) for GPU presentation. Its compute pipelines compile
asynchronously at creation. WebGPU on HTTPS/localhost is required; WebGL is
not a fallback. An outdated surface is reconfigured at the next render boundary,
and a lost surface is recreated. Device loss or uncaptured GPU errors place the
renderer in an explicit failed state; the host displays the error and asks for
reload. Automatic device and asset reconstruction remain future work.

Temporal antialiasing (`temporal.rs`, `temporal.wgsl`) renders GPU-presented
display frames into an sRGB scene target and an `Rg16Float` motion target with
a Halton sub-pixel jitter, culling with the same jittered projection. Culling
alternates between the slot pose buffer and a second one of equal capacity, so
the previous frame's poses stay readable; visible IDs of moving slots carry
`MOVED_SLOT` and only those read their previous pose. One full-screen pass then
blends with the reprojected history (YCoCg neighborhood clamp, Catmull-Rom
history filter) and writes the canvas and the next `Rgba8UnormSrgb` history.
Targets follow the surface size and are destroyed when replaced; pipelines are
created when the feature is enabled. A profiling capture times the resolve as
`taa_resolve`.

Dependencies: core snapshot contracts, `glam`, `bytemuck`, and target-specific
`wgpu`, `web-sys`, and `wasm-bindgen`; Naga is a shader-validation test dependency.
Wgpu usage follows the pinned [30.0.1 API](https://docs.rs/wgpu/30.0.1/wgpu/).
