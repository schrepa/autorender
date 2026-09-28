# Decision 044: Virtual geometry

Status: first two stages adopted. Offline cluster hierarchies, their file
format, GPU cluster selection, forward drawing and cluster shadows work on
retained frames, both GPU-presented and CPU-prepared. Cluster occlusion, a
visibility buffer, streaming and scene authoring come later (see Stages).

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
- Pages fill from the coarsest level down, so streaming can keep the coarse
  levels resident.

`VirtualMesh::decode` validates every section and reference:

- the hierarchy is a forest whose depth matches its header;
- each cluster belongs to exactly one group;
- page, vertex and local-index references are in bounds;
- geometry is finite.

Renderers then upload the sections unchanged.

`Mesh::from_virtual(geometry, fallback_triangles)` makes an ordinary mesh
from the most detailed cut with at most `fallback_triangles` triangles
(`VirtualMesh::cut`, `fallback`). Its bounds hold every level. Every path
that does not draw clusters draws this fallback: frames rendered from a
bare snapshot, transparent batches, and reflection probe captures.

### Settings

The `VirtualGeometry` resource is inserted by default and holds `enabled`
and `pixel_error`. `pixel_error` is the largest simplification error a drawn
cluster may show, in pixels of viewport height. It defaults to 1.5 and is
sanitized into [0.25, 64]. Removing or disabling the resource draws the
fallback meshes, and snapshots carry its sanitized values.

### Selection and drawing

`engine_render`'s `virtual_geometry.rs` and `virtual_geometry.wgsl` keep an
arena of every resident virtual mesh. At upload, its nodes, groups and
clusters are rebased to absolute indices, and its pages are copied
unchanged. The arena is rebuilt when meshes become resident or retire.

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
instance: they draw in the first phase and fill the depth pyramid.

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

### Capacity

Following decision 029, the node queues, the group list and the bin regions
start small. They double after a frame whose read-back counts show dropped
entries, as far as the device's storage binding size allows. A frame with
dropped entries still draws everything that fit, and reports the drops.

## Observable behavior

- `rendering.virtual_clusters` and `rendering.virtual_triangles` count what
  the camera selected, `rendering.virtual_shadow_clusters` and
  `virtual_shadow_triangles` what the shadow maps selected, and
  `rendering.dropped_virtual` counts entries dropped for want of room. All
  are read back a few frames late. Profiles time `virtual_selection` and
  report the arena and frame buffers as `memory.gpu_virtual_bytes`.
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
  street-level maps selected 1.6 million triangles. Frame rates have not
  yet been benchmarked.

## Consequences

- Each triangle's three corners are fetched and transformed separately.
  Clusters with fewer than 128 triangles pay for collapsed corners, so
  vertex work can reach about three times that of an indexed draw of the
  same triangles. Full clusters keep the waste small. Compacting the
  selected triangles into an index buffer could remove it later.
- Rebuilding the arena uploads every resident mesh again. That is fine at
  load time; streaming will replace it.
- Files hold every level. The boulder takes 9.2 MB, about 28 bytes a
  source triangle.
- Selection costs one dispatch per hierarchy level plus four more each
  frame, for every view together.
- Shadow maps redraw virtual casters every frame, and each cascade redraws
  what it covers: their triangles run one to three times the camera's.

## Stages

1. Offline hierarchies, the format, GPU selection, forward drawing and the
   dense scene (this record).
2. Cluster shadows in the cascades and the moving map (this record).
3. Cluster occlusion: two phases against the depth pyramid of decision 040,
   plus a sub-pixel cluster test.
4. A visibility buffer: rasterize instance, cluster and triangle IDs, then
   shade each material in its own pass.
5. Streaming: coarse pages stay resident, the GPU requests pages through
   read-back, and a fixed pool evicts the least recently used pages under a
   byte budget.
6. Authoring: `cargo xtask asset import` builds `EVG1` files for meshes over
   about 500 triangles, a scene section references them, and agents get a
   control, observations and a smoke check.
