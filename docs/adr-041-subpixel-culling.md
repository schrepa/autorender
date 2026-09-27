# Decision 041: Sub-pixel culling

Status: adopted for GPU-presented frames, on by default.
`BrowserApp::set_subpixel_culling(false)` (Stress Lab `?subpixel=off`)
turns it off.

## Context

Seen from afar, many instances shrink below a pixel. The rasterizer samples
pixel centers, so a triangle whose screen rectangle holds no pixel center
produces no fragment, yet its vertices and setup still cost the opaque pass.
From the Stress Lab Metro's aerial overview, about 6,600 frustum-visible
instances (12% of the submitted triangles) cover no pixel center at all.

## Decision

GPU-presented frames that cull without an instance budget test every
frustum-visible instance after culling: the screen rectangle of its posed
mesh bounds (the eight projected corners), widened by a sixteenth of a
pixel, must hold a pixel center on both axes, or the instance is left
undrawn. The margin covers the rasterizer's snapping of vertices to its
subpixel grid and rounding differences between this projection and the
vertex shaders'; boxes reaching the eye plane, lying far outside the
viewport or not finite are always drawn. Skinned slots use their palette
bounds (decision 036). The test uses the projection the frame rasterizes,
jittered on temporally antialiased frames.

Every triangle of such an instance lies within its bounds, so it could
produce no fragment: output stays pixel-identical. Undrawn instances still
count as visible, cast shadows and take part in occlusion culling's next
frame (decision 040) as before. CPU-presented frames, public snapshots and
frames with an instance budget draw them, as before.

## Observable behavior

- `rendering.visible_instances` still counts frustum-visible instances;
  `rendering.subpixel_instances` (read back a few frames late) counts those
  left undrawn, and profiles count them as `render.subpixel_instances`.
- Stress Lab Metro, Firefox on the laptop's integrated GPU, 1512x792 canvas:
  the aerial overview went from 51.4 to 52.1 fps (opaque pass 9.0 to 8.6
  ms, 4,190 instances left undrawn; others that small were already
  occluded). The street-level view is unchanged: the 3,560 instances it
  leaves undrawn were hidden anyway.

## Consequences

The test projects eight corners per frustum-visible instance in the culling
pass, a small part of its cost. Instances just at the size of a pixel may
cover a center on one frame and none on the next; either way they draw what
they would have drawn.
