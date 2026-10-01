# Decision 046: Shared mesh buffers and sorted draws

Status: adopted. Meshes live in shared GPU buffers, indirect draws name
their visible-ID regions as first instances where devices allow it, and
opaque draws and shadow casters draw in state order. Nothing public
changes.

## Context

Every mesh had its own vertex, extras, skin and index buffers, and every
draw bound them, its material group and its region of the visible-ID
buffer. On the Stress Lab's Metro preset (90,000 objects) a frame recorded
about 1,650 indirect draws and 6,800 state calls around them: 3,918 vertex
buffer bindings, 1,655 index buffer bindings, 903 bind groups and 323
pipelines. Browsers charge each call: they serialize it, validate it in the
GPU process and translate it to the native API. Batches were ordered by
mesh, so consecutive draws changed pipeline and material almost every
time.

## Decision

### Shared buffers

`MeshArena` (`crates/engine_render/src/mesh_arena.rs`) keeps one index
buffer and three classes of vertex buffers: positions alone; positions and
extras (vertex colors and second texture coordinates, decision 034); and
positions, extras and joints (decision 036). A class's streams share one
vertex numbering. A mesh takes a range of indices and a range of its class's
vertices (first fit, released ranges coalesce), and its draws carry its
`first_index` and `base_vertex`, which direct draws pass and indirect
arguments hold (the batch records of GPU presentation grew to 64 bytes and
the shadow passes' argument templates take them). A full buffer doubles,
copying its contents on the GPU in a submission of its own before later
writes; a mesh the device's `max_buffer_size` leaves no room for keeps
buffers of its own. The layout changes only when meshes are placed or
retired, and argument templates are rewritten then.
`memory.gpu_mesh_arena_bytes` reports the arena's capacity.

### First instances

The renderer requests `indirect-first-instance` when the adapter offers it.
Indirect arguments then name each draw's visible-ID region as its first
instance (batch `start`, or `map * capacity + start` in the shadow maps),
and passes bind each ID buffer once. Without the feature, first instances
stay zero and every draw binds its own region as before (decision 014).
No mesh vertex shader reads `instance_index`, so draws see the same IDs
either way.

### State order

Opaque batches draw by pipeline variant, then material, then vertex class,
then plan position (`draw_order`, rebuilt each frame from the plan); the
transparent pass keeps the plan's order. Retained shadow casters draw by
pipeline (masked, skinned), vertex buffers and material. Shadow maps store
depth alone, so their order changes no texel. Opaque order decides only
which of two fragments at exactly equal depth wins; Metro captures and the
smoke checks match the previous order pixel for pixel. A pass records what
it has bound (`Bindings`: vertex slots, the index buffer, bind groups
without dynamic offsets) and skips repeats.

## Consequences

On Metro a frame now records 49 vertex buffer bindings, 8 index buffer
bindings, 192 bind groups and 75 pipelines around the same indirect draws.
Most remaining calls are the draws themselves, one per draw group and pass
(WebGPU has no multi-draw indirect).

Measured on Firefox with an Intel Xe-LPG at 1512x792, Metro overview,
three alternating runs against the previous commit (which lacked the
earlier CPU changes of this series too): 45.0-45.8 fps before and
47.2-47.7 fps after, CPU frame time 13.3-14.6 ms before and 10.4-11.0 ms
after, the opaque pass 10.2-10.5 ms on the GPU before and 9.6-9.9 ms after,
and shadow maps 3.0-3.1 ms before and 2.7-2.8 ms after.

Meshes placed together are retired separately, so ranges fragment; first
fit reuses them, and nothing compacts the arena.

## Alternatives and evidence

Render bundles (tried 28 September 2026). GPU presentation writes every
indirect draw's arguments, so the opaque phases and the shadow maps record
the same commands from frame to frame. Recording them into a list compared
by resource identity (wgpu never reuses one) and replaying a cached bundle
drew identical pixels, but once draws are state-sorted browsers charge
little per call: 1,400 indexed indirect draws with 350 bind group changes
cost about 0.08 ms of script time per pass in Firefox and 0.05 ms in Chrome
(0.13 ms with the discrete GPU), against 0.01 ms for a bundle. Metro's
command encoding, about 0.6 ms a frame, did not fall: recording and
comparing the commands cost what the calls saved. Bundles were not adopted.

Merged, vertex-pulled draws (tried 1 October 2026). Engines that draw many
tiny meshes report a fixed cost per instance, which merging them into one
non-instanced draw per material removes (id Software's merged index buffers
in Doom Eternal; AMD's measurements of vertex pulling against instancing).
Metro's impostor cards (decision 048) are that case in its purest form:
about 14,000 instances of six vertices each in one instanced draw. Drawn
instead as one non-instanced draw of six vertices per card, each vertex
carrying its card's visible ID, they took the same time: `opaque_meshes`
measured 8.42 ms against 8.43 ms over three alternating Firefox runs on the
integrated GPU, with identical images. At the 50 ns per instance those
reports suggest, the cards alone would have saved about 0.7 ms. Instancing
costs nothing measurable here, so merging small instances and shadow
casters was not built.
