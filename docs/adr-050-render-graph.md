# Decision 050: The frame as a render graph

Status: adopted. The renderer declares each frame's passes in a render graph
that checks them, culls those nothing uses and derives their attachments'
load and store operations. Rendered images are unchanged; agents can read
which passes ran.

## Context

By decision 049 the renderer's frame was one function of about 2,200
lines over a `Renderer` of 97 fields. Two restructuring steps came first:
the frame became a list of stage methods sharing a frame context, and the
renderer's state became per-feature parts (display, camera bindings,
instance storage, mesh pipelines, virtual geometry and visibility buffer
draws, impostors), with one `Cached` helper for every bind group built
from resources of given generations.

The passes still chose their attachments' operations by hand, and nothing
recorded which passes a frame ran or what they exchanged. The opaque pass
stored the depth buffer only when "ambient occlusion, reflections or
shafts, transparency, decals, a second phase or a visibility-buffer
resolve" read it afterwards, a list every new depth reader had to extend.

What render graphs do elsewhere:
- Frostbite's FrameGraph (Yuriy O'Donnell, GDC 2017): passes declare the
  resources they create, read and write in a setup phase; compiling culls
  passes and resources nothing references and computes lifetimes;
  executing allocates transient resources just in time, aliasing their
  memory, and places barriers.
- Unreal's Render Dependency Graph: passes declare parameter structs of
  shader resources, unordered-access views and render targets with load
  actions; the builder culls passes without consumers, derives barriers,
  allocates and aliases transient resources, merges render passes,
  validates, and names GPU events after passes.
- Granite's render graph (Hans-Kristian Arntzen) also reorders passes and
  merges them into subpasses for tile-based GPUs.
- Bevy's render graph orders nodes connected by slots; resources stay with
  the nodes.

What WebGPU leaves to a graph:
- The browser tracks each resource's usage per pass and synchronizes
  passes itself: there are no barriers or layout transitions to place.
- There is no memory aliasing (no heaps or placed resources), so the
  transient aliasing that saves memory elsewhere is unavailable, and there
  are no asynchronous compute queues to schedule.
- Load and store operations stay the application's: every render pass
  attachment clears or loads, and stores or discards.

## Decision

`crates/engine_render/src/render_graph.rs` holds the graph; it owns no GPU
objects and runs in native tests.

Resources. The twenty resources passes share: the canvas; the depth
buffer; the screen effects' scene, glow and surface targets; temporal
antialiasing's motion target and history; the visibility buffer's IDs and
tile lists; transparent accumulation and revealage; retained poses; GPU
presentation's visible IDs and draws; shadow casters and maps; virtual
geometry's selected clusters; light clusters; the depth pyramid; the
screen effects' results; and environment maps and probe cubes. Each has a
lifetime:
- frame: undefined until a pass writes it and not stored after its last
  reader (depth, the effects' targets, IDs and tiles, accumulation, casters,
  light clusters, effect results);
- kept: defined when the frame starts, every write stored (the history,
  poses, visible IDs the next frame's first phase starts from, shadow maps
  with the kept map, selected clusters with their page requests, the depth
  pyramid the next selection retests with, probe cubes);
- output: the canvas, undefined at the start, every write stored.

Passes. Each declares the resources it reads and writes and, for each
attachment, its access:
- `Clear`: it overwrites the attachment (the visibility buffer's IDs,
  accumulation, full-screen composites);
- `Draw`: it draws over what earlier passes drew there, or over the clear
  value when no earlier pass drew it (the opaque phases);
- `Load`: it draws over earlier contents, which must exist (resolves,
  decals, the transparent composite);
- `ReadOnly`: it tests depth without writing it.

Compiling walks the passes backwards and keeps a pass if it writes a kept
or output resource or something a later kept pass reads; an attachment is
stored when a later kept pass reads it (drawing over it counts) or it is
kept or output, and discarded otherwise. Walking forwards, an attachment
clears where no earlier kept pass drew it and loads where one did, and a
pass reading a frame resource no earlier pass wrote fails compilation; the
frame then returns an error naming the pass and resource.

The renderer's frame. `render_graph::plan` declares a frame from its
`FrameShape` (GPU presentation, targets, phases, which features run):
presentation; shadow caster selection; cluster selection; shadow maps;
light binning; each opaque phase with its visibility-buffer passes (IDs,
tile lists, resolve) and the occlusion test between them; decals;
transparent accumulation and composite; the screen effects; the composite
or temporal resolve; and probe captures. Declared order is execution order:
WebGPU synchronizes passes, and the order was chosen deliberately (shadow
maps before the camera, probes after the frame). The renderer keeps one
graph, declares each frame into it after preparing the features, compiles
it, and encodes each kept pass by its `Pass`, binding the attachments from
the frame's views with the derived operations. Inside a pass, its draws,
bind groups and timestamps stay as before; the screen effects' internal
chains (occlusion, bloom, reflections and shaft targets) keep their own
operations. GPU presentation's compute pass is now recorded after the
frame's CPU-side preparation of virtual geometry and the visibility buffer
rather than before it; their queue writes precede the submitted frame
either way.

Reports. `rendering.render_passes` lists the passes the last frame ran, in
order, and `rendering.culled_passes` those it culled.

## Consequences

The derived operations equal the hand-written ones, with one exception:
the glow target is now discarded after its last writer unless bloom or
reflections read it, and the surface target unless reflections do; both
were always stored before. A test over every frame shape the renderer can
produce (16,496 of them) checks that every frame compiles, that no pass is
culled, and that the depth buffer is stored exactly when the old rule
stored it.

No pass is culled today: every feature already prepares only what it
draws. Culling guards later passes, and a new depth or target reader now
declares what it reads instead of extending each producer's rule.

Measured on the Stress Lab Metro (Firefox 157, Intel Xe-LPG, 1512x792,
three alternating runs against the build before the frame was split into
stages): 67.76 fps before and 67.98 fps after, with the same GPU pass times
(opaque meshes 6.24 and 6.27 ms, the whole frame 14.39 and 14.32 ms).
Declaring and compiling about twenty passes a frame did not show in the
CPU profiles, whose frame times vary from 3.3 to 5.6 ms between runs of
either build. Twenty-eight captures in seven configurations (all effects,
some and none, reflection probes, temporal antialiasing, one opaque phase,
and the dense virtual-geometry scene) match the reference build pixel for
pixel, and the smoke checks pass.

## Alternatives

Passes as trait objects with closures, as Unreal's lambdas. A pass needs
mutable access to several of the renderer's parts at once; matching on a
`Pass` and calling the renderer's method keeps borrows simple.

Bind groups built by the graph. The renderer's bind groups live across
frames and are rebuilt only when a buffer grows, a texture is replaced or a
target is resized; `Cached` keys each by the generations it binds. The
graph binds attachments, which change per frame.

Reordering and merging passes. WebGPU has no subpasses to merge into, and
no aliasing or asynchronous queues for a reordering to serve.
