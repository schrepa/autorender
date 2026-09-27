# Decision 024: Sun shafts

Status: adopted for every renderer path; opt-in (`SunShafts`), and only
with sun shadows.

## Context

The atmosphere's height fog (decision 018) scatters light evenly, whether
or not the sun reaches the air: shadows stopped at surfaces, and a low sun
behind towers made no rays.

## Decision

`SunShafts { intensity, anisotropy, distance }` is a resource that is not
inserted by default: inserting it (its default is on) or authoring a scene
environment's `sun_shafts` section turns shafts on; a missing resource or
an intensity of 0 turns them off. They need shadows and read the fog's
density and height falloff as the medium.

At half resolution, each view ray marches 16 steps, offset per pixel by
fixed interleaved gradient noise, through the fog to the depth buffer's
surface or `distance`, adding at each step the sunlight the far shadow maps
(the kept and moving maps of decision 020) let through, dimmed by the fog in
front of it. The result, scaled by the Henyey-Greenstein phase toward the
sun (`anisotropy`) and `intensity`, is blurred over three texels and added
in the sun's exposed color as a screen blend, in the composite or, on
temporally antialiased frames, to what the resolve shows.

## Observable behavior

- Captures, previews and GPU frames shade alike; the pass reports
  `rendering.sun_shafts`.
- The Stress Lab opts in at intensity 1.5; `effects=no-shafts` removes the
  resource.

## Measurements

None yet: a half-resolution pass of 16 comparison lookups per pixel.

## Consequences

Shafts follow the far maps' resolution and the fog's density: light fog
gives a glow around the sun more than distinct rays, dense fog and tall
casters between the eye and a low sun give rays. Local lights do not
scatter.
