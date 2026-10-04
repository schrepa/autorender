# Decision 017: Temporal antialiasing

Status: adopted as an opt-in for GPU-presented browser frames. Extends
decisions 008, 014 and 015. Simulation, captures, previews, inspection and the
exact CPU presentation path are unchanged.

## Context

The renderer draws one sample per pixel. Tens of thousands of small objects,
thin silhouettes and textures sampled without mipmaps alias: edges stair-step,
distant towers turn into speckle, and both shimmer as objects or the camera
move. Multisampling multiplies depth and color samples in the opaque pass, which
is already the most expensive GPU pass at 250,000 objects, and does not touch
texture or shading aliasing. Post-process filters (FXAA, SMAA) cannot recover
sub-pixel detail and leave the shimmer. Temporal antialiasing spreads the
samples over frames instead: each frame renders once, offset by a different
sub-pixel jitter, and blends with earlier frames reprojected to where their
surfaces now are.

## Decision

`BrowserApp::set_temporal_antialiasing(true)` (renderer:
`Renderer::set_temporal_antialiasing`) antialiases ordinary GPU-presented frames
(decision 014). It is off by default; enabling creates its pipelines, disabling
releases its targets. The Stress Lab accepts `?taa=on`.

- **Jitter.** Each frame's projection is offset by a sub-pixel amount from an
  eight-sample Halton (2, 3) sequence, applied to the view-projection that the
  frame rasterizes and culls with. Culling therefore stays conservative for the
  jittered image.
- **Motion.** The opaque pass writes this frame's linear color to an sRGB scene
  target and, per pixel, the offset to the same surface point in the previous
  frame to an `Rg16Float` motion target. The vertex stage transforms each vertex
  by the previous frame's unjittered view-projection and pose; the fragment stage
  divides per pixel and removes the jitter, so a still pixel has zero motion.
- **Previous poses without copies.** Culling writes displayed poses into two pose
  buffers in turn, so the previous frame's poses stay readable. Reading them in
  the vertex stage measured as the costliest part of the motion (see below), so
  culling marks the visible IDs of moving slots (`NODE_ACTIVE` nodes) with a high
  bit, and only those read their previous pose. Other slots display unchanged
  authority rows; their motion is the camera's.
- **Resolve.** One full-screen pass blends the scene with the history at the
  reprojected position (weight 0.1 for the current frame) and writes the canvas
  and the next history (`Rgba8UnormSrgb`) together. The history sample is
  clamped to the box of the pixel and its four edge neighbors in YCoCg, which
  bounds ghosting from disocclusion, changed shading and approximate motion.
  Moving pixels read the history through a five-tap Catmull-Rom filter: by its
  variance, repeated bilinear resampling at this weight would spread moving
  content by about a pixel. Pixels moving less than 1/64 pixel read their own
  texel.
- **Resets.** The history starts over on the first temporal frame, on resize, on
  a new render identity (reset or replaced world), and after any frame drawn
  without temporal antialiasing. After a structure change (slots added or
  removed) or a replaced pose buffer, one frame uses camera motion only, since
  the other buffer's slots may belong to different objects.

## Observable behavior

- Only frames from `BrowserApp::frame` with GPU presentation are temporal.
  Development steps, renders, `present`/`capture` previews (including
  `gpu: true`) and CPU-presented frames (`?presentation=cpu`) are exact and
  never jittered, as were public snapshots (`Renderer::render`) until that
  path was removed on 2026-10-03; captures are bit-identical with the setting
  on or off.
- Converged still views are smoother and stable: in the Chrome smoke test
  (`web/scripts/taa-smoke.mjs`), per-pixel total variation falls from 47.0 to
  26.9 while 8×8 block averages stay within one level of the exact image, and
  two frames 300 ms apart differ by more than 24 levels in about 0.1% of
  pixels. With mipmapped textures (added later) the exact image already
  measures 28.8, temporal frames 24.4, and 42 of 471,064 pixels (0.009%) differ
  by more than 24 levels 300 ms apart.
- A node that stops moving is displayed at its final pose on the next gather;
  that one frame's remaining motion (at most the rest of one tick's interval) is
  treated as camera motion and left to the clamp. Objects added in a frame get
  camera motion for that frame.
- `render.taa`, `render.taa_history` (history read this frame),
  `render.taa_object_motion` (previous poses valid) and
  `memory.gpu_temporal_bytes` are profiling counters; `taa_resolve` is a GPU
  timing sample. Targets use 16 bytes per canvas pixel (35.7 MB at 2262×987),
  plus one pose buffer of the instance capacity (at most 12.6 MB).

## Measurements

Firefox 156 on the 120 Hz display, the integrated GPU of the Core Ultra 9
275HX, profiling builds, harness pinned to performance cores, 2262×987 canvas
(2.23 megapixels), the reference camera. GPU durations are timestamp medians.
See `artifacts/ff-taa-*.json`.

Resolve variants in one session (90,000 moving objects, two page loads each):

| Resolve | GPU (ms) |
| --- | --- |
| As adopted: Catmull-Rom history, YCoCg clamp | 1.38–1.46 |
| Bilinear history | 1.40–1.55 |
| No neighborhood clamp | 1.45–1.48 |
| No history read (scene and motion in, both targets out) | 0.85–1.08 |

The resolve is bound by memory traffic (about 20 bytes per pixel), not by its
filtering or clamping: the sharper filter and the clamp cost nothing
measurable. At 90,000 moving objects the opaque pass grew from 2.19–2.23 ms to
2.37–2.50 ms.

At 250,000 objects (10% moving), reading previous poses for every slot made the
opaque pass 5.95–8.65 ms against 4.68–5.21 ms without temporal antialiasing and
4.92–5.52 ms for a build that skipped the read (four page loads each). With
moving-slot marks, alternating the setting inside one page (two loads, four
rounds each) measured:

| Setting | Opaque (ms) | Resolve (ms) | Compute (ms) | Frames per second while profiling |
| --- | --- | --- | --- | --- |
| Off | 5.02 | — | 1.51 | 80 |
| On | 5.42 | 1.54 | 1.51 | 68 |

At this canvas size the GPU already limits 250,000 objects, so the added
1.9 ms costs frame rate. Holding the setting for 30 s (one capture every 3 s),
the opaque pass drifted from about 5.3 to 7.4 ms while the resolve and compute
passes stayed flat; with the setting off it stayed near 5.5 ms, apart from one
spike both series showed at the same time. That is consistent with the
integrated GPU lowering its clock under the added load, which memory-bound
passes do not feel.

## Consequences

Temporal antialiasing costs about 0.7 ms of GPU time per megapixel for the
resolve, plus the motion target and moving objects' previous poses in the
opaque pass. Games opt in where that fits their GPU budget. Moving content is
somewhat softer than still content, and the clamp cannot remove every ghost of
fast, thin, high-contrast features. Texture mip chains (added after these
measurements) now remove texture aliasing at its source, leaving geometric
edges to the resolve. Rendering below the canvas resolution and upsampling in
the resolve would trade the added cost back, but the opaque pass at 250,000
objects is dominated by vertex work, so it would help less there.
