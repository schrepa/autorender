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

At quarter resolution, each view ray marches 16 steps, offset per texel by
fixed interleaved gradient noise, through the fog to the depth buffer's
surface or `distance`, adding at each step the sunlight the far shadow maps
(the kept and moving maps of decision 020) let through, dimmed by the fog in
front of it. The result is scaled by the Henyey-Greenstein phase toward the
sun (`anisotropy`) and `intensity`, and stored with the length of the ray.

The half-resolution filter of ambient occlusion (decision 021) brings the
shafts up beside the occlusion. Each half-resolution texel weighs the four
quarter-resolution rays around it bilinearly, and by how near their lengths
lie to its own ray's, within a tenth. Light gathered past a silhouette thus
does not spill onto the surface in front of it. Where no ray is near, the
bilinear weights alone remain. The composite reads the occlusion and the
shafts with the same four taps, which blur them over three half-resolution
texels. It adds the shafts in the sun's exposed color as a screen blend, as
does the resolve on temporally antialiased frames.

## Observable behavior

- Captures, previews and GPU frames shade alike; the pass reports
  `rendering.sun_shafts`.
- The Stress Lab opts in at intensity 1.5; `effects=no-shafts` removes the
  resource.

## Measurements

Firefox on the laptop's integrated GPU, Stress Lab Metro overview at a
2262x968 canvas (30 September 2026):

- At half resolution the march took 1.03 ms. Of that, 0.85 ms were the
  shadow lookups: without them it took 0.19 ms. Point lookups instead of
  comparison filtering saved only 0.05 ms.
- At quarter resolution the march takes 0.39 to 0.43 ms. The shared filter
  (`effects_filter`) takes 0.26 to 0.28 ms, about 0.09 ms more than the
  occlusion filter alone did.
- The composite reads one texture fewer. Shafts, occlusion and filter
  together fell from 1.47 to 0.94 ms.
- Chrome captures match the half-resolution shafts' within 5 of 255
  levels. The overview differs by at most 1 level. A skyline facing the sun
  at golden hour and at dusk differs in 4 pixels by more than 3 levels.

## Consequences

Shafts follow the far maps' resolution and the fog's density: light fog
gives a glow around the sun more than distinct rays, dense fog and tall
casters between the eye and a low sun give rays. Local lights do not
scatter.
