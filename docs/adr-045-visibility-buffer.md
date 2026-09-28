# Decision 045: Visibility buffer

Status: adopted. The camera's opaque virtual geometry clusters (decision
044) shade through a visibility buffer on retained frames, GPU-presented and
CPU-prepared, in both drawing phases. Ordinary meshes, alpha-masked
materials and devices binding too few storage buffers keep forward shading.

## Context

Forward shading runs a mesh's full fragment stage (materials, the sun and
its shadow lookups, local lights, environment and probe reflections) for
every fragment a triangle covers when it draws. Triangles drawn before
nearer ones still pay, and so do the helper invocations filling each 2x2
quad. Virtual geometry keeps triangles about a pixel in size, so most quads
hold one or two covered pixels and a pixel can be shaded several times by
the clusters crossing it. What the camera sees costs a fraction of what the
fragment stage runs.

A visibility buffer separates coverage from shading. Rasterization only
records which triangle covers each pixel, and shading then runs once per
pixel for the triangle that won. WebGPU has no 64-bit atomics, so there is
no software rasterizer writing depth and payload together, but hardware
rasterization into an integer target with the ordinary depth test records
the same thing.

## Decision

### Settings

`VirtualGeometry::visibility_buffer` (on by default) chooses the visibility
buffer; off, clusters shade as they draw (decision 044). Scenes set it in the
environment's `virtual_geometry` section (`visibility_buffer`, omitted when
true), agents read and change it with `inspect` and the paused `environment`
control, and the Stress Lab takes `?visibility=off`. Frames report
`rendering.visibility_buffer`: whether this frame resolved through it.

A frame resolves through it when the setting is on and the device binds
enough storage buffers to a fragment stage: 8, or 11 on frames with local
lights (the renderer now requests the adapter's
`max_storage_buffers_per_shader_stage`). Of the camera's bins, those of
alpha-masked materials still draw forward, since the visibility pass tests no
coverage; all others resolve.

### Passes

Each drawing phase (decision 040) draws its ordinary batches, and bins that
draw forward, as before. For bins that resolve, three more passes follow.

1. **Visibility.** The phase's selected clusters draw into an `Rg32Uint` ID
   target (cleared to 0) with the shared depth buffer, depth-tested and
   written as forward clusters would be (`vs_visibility`, `fs_visibility`).
   A pixel keeps its entry in the selected clusters plus one, and its
   triangle | bin << 7. Positions are computed as `vs_virtual` computes them,
   so the depth buffer, the depth pyramid and every later pass see the same
   surfaces.
2. **Classification.** A compute pass (`visibility.wgsl`) reads the target in
   8x8 tiles and appends each tile once to the list of every bin with a
   pixel in it; a second dispatch writes each bin's draw arguments. Each
   bin's list can hold every tile of the viewport, so no tile is ever left
   out; where that would exceed the device's storage binding size, the frame
   draws its bins forward.
3. **Resolve.** Each bin draws two triangles per listed tile with its
   material's resolving pipeline (`vs_resolve`, `fs_resolve*` for the
   canvas, screen-effect and temporal targets, lit or not). Pixels of other
   bins or of nothing discard; the rest shade as below and write every
   target a forward fragment writes. Depth is read-only, and after the last
   phase the sky draws in the same pass.

Ordinary meshes draw before each phase's clusters, and phase one resolves
before phase two draws anything, so every resolved pixel shows its phase's
nearest surface and ordinary meshes drawn later in phase two cover it
correctly. Phase two clears the target and resolves only its own clusters.

### Shading a pixel

A resolved pixel reads its selected cluster (visible ID and cluster), poses
its triangle's three corners with `transform_vertex` exactly as forward
vertices are posed, and interpolates their outputs at the pixel center with
perspective-correct barycentric weights from the corners' clip-space edge
functions (Olano and Greer). The same formulation holds for triangles
crossing the eye plane. The counter-clockwise test of the same determinant
tells faces toward the camera, for double-sided materials.

Neighboring pixels may show other triangles, so derivatives cannot come
from the quad. The resolve evaluates the same triangle's weights at the
pixel's partner in its 2x2 quad, across and down, as fine derivatives take
them. From these come the derivatives of the world position (shadow receiver
planes, the normal map's tangent frame) and of the texture coordinates:
every map is sampled with `textureSampleGrad` through its coordinate
transform. Specular antialiasing needs how the shading normal turns. The
resolve rebuilds the shading normal at both partners, geometric normal,
normal map sample and ripples included, and widens the lobe by the
difference as forward fragments do.

`meshes.wgsl` shares the rest with forward fragments: `material_texels` (implicit
derivatives) and `material_texels_grad` sample the same maps; `textured_point`,
`shading_normal` and `widened` build the surface point; `finish`,
`local_light`, `scene_targets` and `temporal_targets` light and write it.
Forward shading computes exactly what it did before.

Temporally antialiased frames need each pixel's motion. The resolve
interpolates the mesh-space point, poses it by the slot's previous pose when
the slot moved, and projects it with the previous frame's view projection,
as `vs_virtual_temporal` does per vertex.

### Resources and capacity

`visibility.rs` owns the ID target (8 bytes a pixel), the tile lists (4
bytes per tile per bin), each bin's tile count and draw arguments, and a
256-byte uniform per bin and phase (its first selected cluster, bin, list
start and tiles across) that the passes bind with dynamic offsets. They
follow the viewport's size and the frame's bins, and profiles count them in
`memory.gpu_virtual_bytes`. Following decision 029, only device limits bound
them, and the frame falls back to forward shading rather than dropping
tiles.

## Observable behavior

- The virtual geometry smoke check's street-level and ground-level views of
  the dense scene resolve on CPU-prepared and GPU-presented frames alike,
  and their captures match exactly. Against forward shading of the same
  views, 13 and 17 of 471,064 pixels differ by more than 8 levels.
- Imported glTF materials with every map (a local test of Khronos'
  DamagedHelmet sample: base color, metallic-roughness, normal, occlusion and
  emissive maps) differ from forward shading in 0.04% and 0.085% of pixels
  at two views, on thin normal-mapped highlights. Double-sided, seen from
  inside, they differ in 0.06% and 0.11%. CPU and GPU captures of the
  resolve match exactly.
- On an Intel Xe-LPG integrated GPU (Chrome, vsync off, 1400 by 900), the
  dense scene's median frame took 8.8 to 9.7 ms resolving against 9.5 to 9.9
  ms forward at street level, and 8.2 to 9.0 against 8.4 to 9.2 ms from
  above: within noise. Its materials are plain colors, so little shading was
  saved; the extra passes cost about as much.
- `rendering.visibility_buffer` is false with the setting off, without
  virtual geometry, or where the device limits the resolve.

## Consequences

- Frames that resolve end the opaque pass, draw the visibility pass, run the
  classification and draw the resolve pass in each phase: more pass
  boundaries, 8 bytes a pixel of target, and one more read of it per bin
  whose tiles overlap.
- Each resolved pixel fetches and poses three corners and solves its
  weights three times; shading runs once per pixel instead of once per
  covered fragment and helper.
- Resolved derivatives follow fine derivatives of forward fragments, so
  texture levels and specular antialiasing can differ from forward shading
  in a few pixels where hardware takes coarse ones.
- Alpha-masked materials still draw forward. Testing coverage in the
  visibility pass would need their textures there.
- Only the camera's virtual geometry resolves. Ordinary meshes could resolve
  too once their instances and indices are addressable per pixel, and
  heavier shading (more lights, reflections, glTF extensions) gains more.
