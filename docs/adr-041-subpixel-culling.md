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
frame (decision 040) as before. CPU-presented frames and frames with an
instance budget draw them, as before (as did public snapshots until
2026-10-03).

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

## Alternatives and evidence

Culling back faces before the rasterizer (tried 28 September 2026). On
Metro 67% of instance triangles lie in groups facing along one of their
mesh's axes, about half of which face away from any view. WebGPU has no
multi-draw indirect and an empty indirect draw costs about 0.3 µs on the
laptop's integrated GPU, so drawing each face group on its own would add
six draws per draw group and phase. The variant without extra draws was
measured: each mesh's downward faces (all three vertices at one height,
winding facing down) last in its indices, a pass testing the eye's height
in every visible instance's mesh space, and indirect draws shortened past
those faces for draw groups the eye sees only from above. Output was
identical and the draws shortened, but the opaque pass measured 9.39 ms
with it and 9.31 ms without (one build, Firefox) while the test added
0.23 ms: the rasterizer rejects back faces cheaply. Not adopted.
