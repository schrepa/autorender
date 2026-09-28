# Decision 040: Occlusion culling

Status: adopted for GPU-presented frames, on by default.
`BrowserApp::set_occlusion_culling(false)` (Stress Lab `?occlusion=off`)
turns it off.

## Context

Frustum culling (decision 014) keeps everything inside the view. In a dense
city most of that is hidden: from the Stress Lab Metro's street-level view,
92% of the 46,600 instances inside the frustum are behind nearer buildings,
and from its aerial overview about a quarter. Their vertices and small
triangles still cost the opaque pass, which bounds such scenes.

## Decision

GPU-presented frames that cull without an instance budget draw the opaque
pass in two phases around an occlusion test, entirely on the GPU:

1. **Phase one** draws every frustum-visible instance the last frame's test
   did not find hidden, and every instance occlusion never hides: meshes with
   joints (their vertices leave their bounds, decision 036) and transparent
   draw groups (which draw in the transparent pass).
2. A **depth pyramid** of phase one's depths: level 0 holds the farthest depth
   of each 2x2 block of the depth buffer, each further level the farthest of
   each 2x2 block of the level below (the smallest values: depth is reversed,
   [decision 042](adr-042-reversed-depth.md)). Levels have power-of-two
   sizes, so each halves the last exactly.
3. The **test** projects every other frustum-visible instance's posed mesh
   bounds and compares their nearest depth with the farthest depth of the
   pyramid texels (at most 2x2, at the finest level that small) its screen
   rectangle touches. A box is hidden only beyond that depth by more than
   depth-buffer steps and projection rounding (a relative margin of 2^-16 of
   the depth); boxes reaching the eye plane or
   too large for the pyramid are not. The result marks each tested instance
   for the next frame's phase one.
4. **Phase two** draws the instances phase one skipped that the test does not
   hide, from their own compacted IDs and indirect arguments, then the sky.

The test only hides instances behind what this frame drew, so frames draw
exactly what they would without it: captures of GPU-presented frames match
the CPU reference pixel for pixel. Only which phase draws an instance
changes; near-coplanar instances keep drawing in phase one, in the same order
as without the test. CPU-presented frames, public snapshots and frames with
an instance budget draw in one phase, as before. Shadow maps select casters
without it: hidden instances still cast shadows.

## Observable behavior

- `rendering.visible_instances` still counts frustum-visible instances;
  `rendering.occluded_instances` (read back a few frames late) counts those
  the test left undrawn.
- Profiles add the `occlusion` GPU sample (the pyramid, the test and phase
  two's compaction); `opaque_meshes` spans both phases.
- Stress Lab Metro, Firefox on the laptop's integrated GPU, 1512x792 canvas:
  the street-level view went from 41.6 to 50.6 fps (opaque pass 11.4 to 6.1
  ms, 42,871 of 46,603 visible instances hidden); the aerial overview from
  47.9 to 48.5 fps (21,900 of 86,700 hidden).

## Consequences

The pyramid, test and second compaction cost about 0.5 ms of GPU time per
frame, and the second phase repeats the opaque pass's draw commands, most of
them empty. An instance that becomes visible draws in phase two for one
frame, after the others: where it exactly overlaps another surface at equal
depth, that frame may show the other. Views that hide little gain little.
