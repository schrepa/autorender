# Decision 042: Reversed floating-point depth

Status: adopted for every camera projection and the main depth buffer.

## Context

A standard projection maps the near plane to depth 0 and the far plane to 1,
and `1/z` spends almost all of a depth buffer's values close to the camera.
With a 24-bit buffer, the Stress Lab's 0.1 m near plane and 5 km far plane
resolve about 2 mm at 60 m, 5 cm at 300 m and 60 cm at 1 km: surfaces closer
together than that fight pixel by pixel. The Metro's billboards, whose
pictures sit 1 mm in front of their boards, turned black beyond about 40 m.

Reversing depth (near 1, far 0) on a floating-point buffer cancels that
curve: floats are densest near 0, exactly where `1/z` compresses the far
distance, so resolution stays about proportional to distance. Measurements
(Nathan Reed, "Depth Precision Visualized") find essentially no precision
errors with it, and it is the common choice: Godot 4.3 switched to it, Unity
uses it on Direct3D, Metal and Vulkan, and Bevy and Unreal reverse depth
with an infinite far plane. WebGPU clips depth to [0, 1] and always offers
`depth32float`, so it needs no extension. Logarithmic depth, the other
option, writes depth from fragment shaders and so disables early depth
tests, which the opaque pass relies on.

## Decision

- `Camera::view_projection` maps the near plane to depth 1 and the far plane
  to 0 (glam's right-handed perspective with the planes swapped). The far
  plane stays: objects beyond `far` still clip.
- The main depth buffer is `Depth32Float`, clears to 0, and depth tests keep
  the greater value; the sky draws on the far plane at 0 where the depth is
  still cleared.
- Everything reading depth follows the convention (`NEAR_DEPTH` 1 and
  `FAR_DEPTH` 0 in the renderer): the sky tests of ambient occlusion,
  reflections and decals (`depth > 0`), the reflections' behind-surface test,
  the occlusion pyramid's farthest depth (the smallest value) and its test's
  margin, now relative (2^-16 of the depth) so it stays tight at every
  distance, the near shadow maps' view fits, and the light clusters' planes.
- Shadow maps keep their own orthographic depth, uniform across their range.

## Observable behavior

- Surfaces a millimeter apart stay apart at every distance in the Metro; its
  billboards show their pictures from afar. Exactly coplanar faces still
  split by rounding, as with any depth buffer: models must not overlap them.
- Occlusion culling hides more distant instances (Metro overview: 20,140 to
  23,900).
- Firefox on the laptop's integrated GPU, Metro, same run: the overview went
  from 47.4 to 48.6 fps and the street-level view from 53.1 to 54.2 fps.

## Consequences

Custom shaders that read depth must treat larger values as nearer and 0 as
the sky. Pixels change where surfaces used to fight, and slightly where
screen effects rebuild positions from the more precise depth.
