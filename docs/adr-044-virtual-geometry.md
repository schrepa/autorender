# Decision 044: Virtual geometry

Status: five stages adopted. Offline cluster hierarchies, their file
format, GPU cluster selection, forward drawing and cluster shadows work on
retained frames, both GPU-presented and CPU-prepared; GPU-presented frames
also cull occluded and sub-pixel clusters. Scenes reference virtual meshes,
the importer builds them, and agents control and observe them. Hosts load
each file's coarse prefix, and pages stream in as views need them. A
visibility buffer comes later (see Stages).

## Context

Every triangle of a mesh costs vertex work wherever the mesh is. A scanned
rock of 300,000 triangles draws all of them at the far end of a field, where
it covers twenty pixels, and draws its far side up close. Authored levels
of detail need artists, pop when they switch, and choose one level for the
whole object, so a mesh near the camera still draws its distant half at full
detail. The opaque pass already bounds large Stress Lab scenes, and it is
vertex-bound in the city.

Nanite-style virtual geometry splits a mesh into clusters of about 128
triangles. Groups of clusters are simplified into the next level's clusters,
forming a hierarchy. Each view picks the clusters whose simplification
error projects below a pixel threshold, so triangles stay about a pixel in
size at any distance, with culling per cluster. Browsers constrain how far
this goes: WebGPU has no mesh shaders, standard multi-draw indirect or
64-bit atomics, so there is no software rasterizer with atomic depth and
payload. It does have compute shaders, indirect draws and dispatches, and
storage buffers, with eight storage buffers per stage by default.

## Decision

### Offline hierarchies

`tools/vgbuild` (crate `engine_vgbuild`) builds hierarchies with the
vendored meshoptimizer 1.3 `clusterlod` (`vendor/meshoptimizer`). It is
compiled only into native tools; the browser never runs it.

- Clusters hold at most 128 triangles over at most 255 vertices.
- Groups of neighboring clusters are simplified together into the next
  level's clusters. Each group records its simplification's bounding sphere
  and error; both only grow up the hierarchy.
- Vertex normals, and texture coordinates where present, steer
  simplification; texture seams are protected.
- Open borders are locked at every level, so edges that meet another mesh
  (terrain tiles) never crack.
- Each level's groups get a bounding volume hierarchy of eight children per
  node. A node's error is the largest in its subtree.
- Each cluster gets a culling sphere and a normal cone
  (`meshopt_computeClusterBounds`).
- Meshes whose every edge joins exactly two triangles are flagged closed
  (`FLAG_CLOSED`, `VirtualMesh::closed`).

### The `EVG1` format

`engine_assets::VirtualMesh` reads files of little-endian 32-bit words:

- a 32-word header;
- nodes (8 words each), groups (8 words each) and clusters (12 words each);
- a page table (2 words a page), then pages of at most 128 KiB.

Pages hold each cluster's vertices and triangles.

- A stored vertex is four words: 16-bit position steps on the mesh's grid,
  an octahedral normal (decision 032) and 16-bit texture coordinates over
  the mesh's range.
- Triangles are 8-bit local indices, four to a word.
- Groups simplified with more error than the file's pin error are pinned:
  the pin error is the smallest whose cut holds at most 4,096 triangles
  (`PINNED_TRIANGLES`). Pinned groups fill the first pages, and the other
  groups follow from the coarsest level down.
- The header (format version 2) records the pinned pages, the pin error and
  the prefix length: the metadata sections and the pinned pages, all a
  renderer needs before any other page arrives. `VirtualMesh::prefix_bytes`
  reads that length from a file's first 128 bytes.

`VirtualMesh::decode` accepts a whole file or its prefix, and validates
every section and reference:

- the hierarchy is a forest whose depth matches its header;
- each cluster belongs to exactly one group;
- errors grow toward coarser groups, and pinned groups lie in pinned pages;
- page, vertex and local-index references are in bounds, for the pages it
  holds;
- geometry is finite.

`VirtualMesh::validate_page` checks a page's words the same way when they
arrive. Renderers upload the sections unchanged.

`Mesh::from_virtual(geometry, fallback_triangles)` makes an ordinary mesh
from the most detailed cut within the pinned pages with at most
`fallback_triangles` triangles (`VirtualMesh::cut`, `fallback`), so a prefix
suffices. Its bounds hold every level. Every path that does not draw
clusters draws this fallback: frames rendered from a bare snapshot,
transparent batches, and reflection probe captures.

### Settings

The `VirtualGeometry` resource is inserted by default and holds `enabled`,
`pixel_error` and `max_resident_pages`. `pixel_error` is the largest
simplification error a drawn cluster may show, in pixels of viewport height.
It defaults to 1.5 and is sanitized into [0.25, 64]. `max_resident_pages` is
an optional budget of streamed pages (see Streaming), none by default and at
least 1. Removing or disabling the resource draws the fallback meshes, and
snapshots carry its sanitized values.

### Authoring and agents

- **Scenes.** Version 8 meshes take `virtual` sources:
  `{"kind": "virtual", "uri": "rock.evg", "fallback_triangles": 4096}`
  (the fallback optional). Documents list the file as a virtual mesh
  (`SceneFileKind::VirtualMesh`, `virtual_mesh` in the web bindings' file
  list). Hosts fetch its first 128 bytes and then its prefix by byte range
  (`virtual_prefix_bytes`), and supply the prefix with the file's URL
  (`SceneFiles::add_virtual_mesh`, `SceneFileSet::add_virtual_mesh`, as
  `loadSceneFiles` does); renderers stream the other pages from that URL. A
  whole file works too, from a server that ignores ranges or as a buffer.
  `SceneFiles` decodes and validates it on instantiation. The environment
  takes a `virtual_geometry` section (`enabled`, `pixel_error` within
  [0.25, 64], an optional `max_resident_pages` of at least 1); omitted, a
  document's environment turns virtual geometry on.
- **Import.** `cargo xtask asset import` builds `EVG1` files for opaque
  primitives of at least 512 triangles without vertex colors, second
  texture coordinates or joints (`--virtual-triangles N`,
  `--no-virtual-geometry`). Alpha-tested ones stay mesh buffers because
  virtual shadows cast whole clusters. The development server serves `.evg`
  files under `/scenes/`, answering single byte ranges.
- **Agents.** `describe` lists `VirtualGeometry`, `inspect` reads it as its
  scene section, and the paused `environment` control replaces it for the
  session. Before capturing a view at full detail, present frames until
  `rendering.virtual_missing_pages` and `virtual_loading_pages` stay 0 (see
  Observable behavior). `web/scripts/virtual-geometry-smoke.mjs` checks the
  dense scene: loading fetches prefixes only, pages stream until views lack
  none, GPU-presented and CPU captures match, clusters draw and cast
  shadows, a budget bounds resident pages, pixel error and disabling take
  effect, and invalid settings are refused.

### Selection and drawing

`engine_render`'s `virtual_geometry.rs` and `virtual_geometry.wgsl` keep an
arena of every resident virtual mesh. At upload, its nodes, groups and
clusters are rebased to absolute indices, and its pages load unchanged into
slots as Streaming describes. The arena is rebuilt when meshes become
resident or retire.

Each frame, one compute pass selects clusters for every view: the camera's
and those of the shadow maps drawn every frame (see Shadows below). A view
takes the instances of batches whose mesh has virtual geometry and whose
material is opaque:

1. **Seeding.** The camera's instances are the GPU-compacted visible IDs on
   GPU-presented frames and the slot IDs on CPU-prepared ones; a shadow
   map's are the casters its selection chose. Each one seeds the root of
   every level into a node queue, tagged with its view.
2. **Hierarchy walk.** The hierarchy is walked one level per indirect
   dispatch, with two queues taking turns as input and output. A node stops
   the walk when its sphere lies outside the frustum or its largest error
   projects at or below the threshold. A leaf passes its group on.
3. **Cluster tests.** A group draws each of its clusters whose own error
   projects at or below the threshold. A cluster's own error is that of the
   finer group it was simplified from. A drawn cluster must also have its
   sphere in the frustum and a normal cone that does not face away from the
   eye. The cone test is skipped for double-sided materials and for poses
   that scale unevenly; mirroring poses flip the cone.

An error `e` of a sphere at distance `d` from the eye, of radius `r`, projects
to `e / max(d - r, near) * cot(fovy / 2) / 2`. That is a fraction of the
viewport height, compared with `pixel_error / height`. The walk and the
cluster tests use the same spheres, whose errors only grow up the hierarchy.
So each point of the surface is drawn by exactly one selected cluster, with
no cracks or overlaps. Pose scale multiplies spheres and errors.

Selected clusters append a visible ID and a cluster to their material's bin,
a region of one buffer. The first drawing phase of the opaque pass draws
each bin with one `draw_indirect` call:

- 384 vertices an instance, three for each triangle of the largest cluster;
- one instance per selected cluster, from an instance-stepped stream of
  those pairs.

`meshes.wgsl`'s `vs_virtual` and `vs_virtual_temporal` read the local index
and the stored vertex from the pages and dequantize them. They move the
corners past a cluster's own triangles outside the view. Fragment stages
are those of ordinary meshes. Received shadows, local lights, reflections,
ambient occlusion and temporal antialiasing all apply unchanged. On
temporally antialiased frames, moving instances read their previous poses
for motion.

Occlusion culling (decision 040) never tests batches of virtual meshes per
instance; their clusters are tested instead (see Occlusion below).

### Occlusion and sub-pixel clusters

GPU-presented frames that draw two phases (decision 040) test the camera's
clusters, once they pass the other tests, against the depth pyramid:

1. **Selection** compares each cluster's bounds with the last two-phase
   frame's pyramid under the projection that frame rasterized, while that
   pyramid is this one's (the same size, rebuilt no later than the last
   frame). Clusters it hides go to a retest list instead of their bins,
   except when the list is full; the rest draw in phase one and fill this
   frame's pyramid.
2. **The retest**, after phase one and this frame's pyramid, compares each
   waiting cluster with it, as presentation's test compares instances. Those
   it does not hide go to phase-two twins of their bins, drawn after phase
   two's instances.

A cluster is only left undrawn behind what this frame drew, so output
matches frames without the test pixel for pixel: CPU-prepared and
GPU-presented captures of the dense scene match exactly at street level and
from above. Whether a cluster waits depends on the last frame only, so a
wrong guess costs time, never pixels. The same frames leave clusters whose
bounds cover no pixel center undrawn (decision 041's test on their bounding
box). Shadow views skip both tests.

### Shadows

Shadow caster selection (decision 020) puts instances of virtual meshes into
the ID lists of the near cascades and the moving map that they can darken,
as any caster, but never into the kept map. Its view of them would go stale
as the camera's detail changes. Cluster selection then treats each of those
maps as a view:

- **Region.** The map's selection region (light-space rectangle and depth)
  is tightened by the view frustum's slice up to the map's reach, keeping
  the frustum planes that sweeping the slice toward the sun never crosses.
  A caster outside them darkens nothing visible.
- **Detail.** Errors project as the camera sees them, so a map casts the
  surfaces the camera draws, and nothing shadows itself for want of
  matching detail. Distances below the nearest of the map's receivers
  project as that distance: the previous cascade's handover start for a
  cascade, the last cascade's reach for the moving map. Where the camera
  sees none of a sphere, a map also accepts `pixel_error` of its texels.
  Both only loosen for larger spheres farther away, so each cut still
  covers every surface once.
- **Faces.** Closed meshes cast only the clusters whose normal cones face
  the sun, which hide the rest; others cast both faces.

Each map's clusters go to a bin of its own, drawn last in the map's pass by
`vs_shadow` (`virtual_geometry.wgsl`): depth only, flattened onto the map's
near plane, with the maps' constant unit of bias. CPU-prepared frames select
the same way from their listed casters.

### Streaming

Hosts load each virtual mesh's prefix, and renderers load its other pages as
views need them into a pool of 128 KiB slots (`virtual_streaming.rs` keeps
the bookkeeping, `virtual_geometry.rs` the pool and the fetches):

- **Arena.** The pages buffer starts with a slot table, one word per page of
  every resident mesh (its slot, or none), followed by the pool's slots.
  Vertex shaders find a cluster's words through its page's slot. The groups
  buffer ends with one residency bit per group.
- **Residency.** A group can draw when every page holding its clusters is
  resident and so is every coarser group holding clusters made from it.
  Selection treats a group that cannot draw as absent: none of its clusters
  draw, and the coarser cluster made from it draws instead of refining. Pinned pages are always resident, so each surface is drawn
  exactly once at whatever detail is resident, never with holes.
- **Requests.** Selection records two bitsets after its counters: the
  groups it drew from and the groups it wanted but could not draw from.
  They are read back with the counters, a few frames late, on GPU-presented
  and CPU-prepared frames alike.
- **Loads.** The CPU marks the pages of the groups drawn from as used at
  that frame. From each wanted group it walks up to the coarser groups it
  also needs, and begins loading their absent pages, coarser first, as many
  as have room. Pages a whole file holds load from memory; the others are
  fetched by HTTP byte range from the file's URL, one request for each run
  of consecutive pages, validated on arrival and written into slots.
  Failed fetches wait 120 frames before they are requested again.
- **Room.** The pool doubles when full, up to the budget and the device's
  storage binding size. Beyond that, the least recently used unpinned page
  unused for 8 frames (longer than read-backs take) makes room, and loads
  finding none are dropped until some is. `max_resident_pages` bounds the
  resident pages; lowering it evicts the least recently used at once, and
  pinned pages stay even beyond it.
- **Rebuilds.** When meshes become resident or retire, the arena is rebuilt:
  the pinned pages load again (every page a whole file holds, without a
  budget), fetches and read-backs of the older layout are ignored, and the
  other pages stream again as views ask for them.

### Capacity

Following decision 029, the node queues, the group list and the bin regions
start small. They double after a frame whose read-back counts show dropped
entries, as far as the device's storage binding size allows. A frame with
dropped entries still draws everything that fit, and reports the drops. The
page pool grows as Streaming describes, with an optional budget.

## Observable behavior

- `rendering.virtual_clusters` and `rendering.virtual_triangles` count what
  the camera drew, `rendering.virtual_shadow_clusters` and
  `virtual_shadow_triangles` what the shadow maps drew,
  `rendering.virtual_occluded_clusters` the camera's clusters the retest
  left undrawn, and `rendering.dropped_virtual` entries dropped for want of
  room. All are read back a few frames late. Profiles time `virtual_selection` and
  report the arena and frame buffers as `memory.gpu_virtual_bytes`.
- `rendering.virtual_resident_pages` and `virtual_loading_pages` count the
  pages resident and being fetched. `virtual_missing_pages` counts those the
  last read-back frame wanted and lacked: 0 once that frame's view had all
  the detail it asked for. `dropped_virtual_pages` counts those of them, and
  of arrivals since, that found no room. Profiles report all four as
  `render.*` counters.
- Loading the dense scene fetches its files' prefixes: 2.0 MB of 50.8 MB,
  each file's metadata and one pinned page. The smoke check's street-level
  and ground-level views then stream 73 more pages (9.5 MB), after which
  their GPU-presented and CPU captures match exactly. A budget of 8 pages
  keeps 8 resident and leaves 48 of the ground view's pages missing (and
  dropped); its captures still match.
- The Stress Lab's dense scene (`scene=dense`, preset "Dense geometry",
  `docs/stress-lab.md`) holds 2,000 objects on 25 terrain tiles:
  - boulders of 327,680 triangles;
  - fluted columns of 591,360;
  - tiles of 524,288.

  That is about 770 million triangles in all. With the columns closed (a
  first stage left their texture seam open, locking it at every level), its
  high overview draws 3,569 clusters (339,870 triangles) and its shadow maps
  4,440 (428,054); street level draws 3,832 (432,418), and its shadow maps
  9,190 (1.02 million) by day and about 8,700 at golden hour. Nothing is
  dropped. Without the tightened regions and the unseen texel error, the
  street-level maps selected 1.6 million triangles. Occlusion leaves 176 to
  354 of about 3,000 camera clusters undrawn at street level and 4 from
  above: the boulders stand apart on open ground. Frame rates have not yet
  been benchmarked.

## Consequences

- Each triangle's three corners are fetched and transformed separately.
  Clusters with fewer than 128 triangles pay for collapsed corners, so
  vertex work can reach about three times that of an indexed draw of the
  same triangles. Full clusters keep the waste small. Compacting the
  selected triangles into an index buffer could remove it later.
- Rebuilding the arena uploads every resident mesh's metadata and pinned
  pages again, and its streamed pages load again as views ask for them:
  fine when scenes load, wasteful if meshes came and went every few frames.
- Files hold every level. The boulder takes 9.2 MB, about 28 bytes a
  source triangle, of which hosts load a 410 KB prefix (the dense scene's
  prefixes run 410 to 633 KB).
- Metadata (nodes, groups, clusters) stays resident for every resident
  mesh; only vertices and triangles stream.
- Detail arrives a few frames after a view first wants it (a read-back, then
  a fetch). Until then coarser clusters draw, so detail sharpens in place;
  a view turning quickly shows coarse surfaces for those frames.
- Selection costs one dispatch per hierarchy level plus four more each
  frame, for every view together.
- Shadow maps redraw virtual casters every frame, and each cascade redraws
  what it covers: their triangles run one to three times the camera's.

## Stages

1. Offline hierarchies, the format, GPU selection, forward drawing and the
   dense scene (this record).
2. Cluster shadows in the cascades and the moving map (this record).
3. Cluster occlusion against decision 040's pyramids and sub-pixel clusters
   (this record).
4. Authoring: scene sources, import, the environment section, the
   inspector and a smoke check (this record).
5. A visibility buffer: rasterize instance, cluster and triangle IDs, then
   shade each material in its own pass.
6. Streaming: pinned coarse pages stay resident, views request pages
   through read-back, and a growing pool evicts the least recently used
   pages under an optional budget (this record).

