# Decision 025: Screen-space reflections

Status: adopted for every renderer path; opt-in (`Reflections`).

## Context

Glossy surfaces reflected only the analytic sky (decision 019): water,
glass and polished floors could not show the city around them.

## Decision

`Reflections { intensity, max_roughness, distance }` is a resource that is
not inserted by default: inserting it (its default is on) or authoring a
scene environment's `reflections` section turns reflections on; a missing
resource or an intensity of 0 turns them off.

While they run, the opaque pass writes each pixel's reflectivity into the
glow target's second channel (which becomes `Rg8Unorm`): the environment
BRDF's share of its light, fading out as roughness approaches
`max_roughness` and through fog. Frames with screen effects also write a
surface target (`Rgb10a2Unorm`): the normal the lighting shaded, with its
normal map and ripples, octahedral, and the roughness. At half resolution,
from every reflective pixel, a ray along the mirror direction of that
normal, up to `distance`, crosses the screen as a straight line in pixels
and depth. It walks a pyramid of the depth buffer's nearest depths (the
hierarchical march of AMD's FidelityFX SSSR): a cell the ray crosses in
front of everything is skipped and the next one is coarser, a cell it may
meet is refined, down to single pixels, so every pixel along the ray near a
surface is visited. The first pixel the ray passes behind is its hit,
weighted by how close behind that surface the ray passes (0.3 world units
plus 5% of its length), and faded toward the screen edges and the ray's
end. A ray heading toward the camera reaches what it meets from the side
the camera cannot see, and crosses much of the world per pixel: for it the
visible surface counts fully through that thickness plus the ray's length
across a pixel there, then fades over a solid object's depth (1 world
unit). Judged like other rays, the undersides of round mirror objects (the
park sculptures) rejected the ground and objects around them in patches
that showed the sky and moved with the camera. A hit reads the light
reflected over the surface's roughness lobe: a cone 1.19 GGX alphas wide on
each side (the Phong lobe matching GGX narrows to 0.244 of its peak there,
as in cone-traced reflections), whose width where it lands picks one of six
half-resolution levels of the scene blurred with the bloom's downsampling
filter; mirrors read the scene itself. Reflections traced along the depth
normal and read sharp drew brushed and rough metal as a flat mirror over
the streaked, blurred sky reflection its lighting showed, so such surfaces
changed look with whatever happened to be on screen. A filter then blends
each half-resolution result with its 3x3 neighbors on the same surface
(depth within 2%), by a tent and by their weights. The composite, or what
the temporal resolve shows, replaces the pixel's reflectivity-weighted
share with the filtered reflected color. Misses keep the sky reflection
already shaded.

Each half-resolution texel traces one pixel of its 2x2 block: the same one
on frames without temporal antialiasing, another each frame with it, so the
resolve's history gathers all four. Reflections were first traced with 24
fixed steps per ray: neighboring rays then met thin objects (fences, poles,
window frames) at some steps and not others, the same way every frame, so
glossy surfaces showed a screen-fixed speckle that temporal antialiasing
could not average and that crawled as the camera moved.

## Observable behavior

- Captures, previews and GPU frames match; the pass reports
  `rendering.reflections`.
- The Stress Lab opts in; `effects=no-reflections` removes the resource.

## Measurements

Firefox on the laptop's integrated GPU, Stress Lab Metro at 1512x792: the
`reflections` sample (the depth pyramid, the blurred scene levels, the
march and the filter) takes 1.07 ms in the overview and 1.36 ms in the
street-level view: 0.73 and 0.97 ms before the surface target and the
roughness cone, and 0.27 ms for the fixed steps first traced. Writing the
surface target added about 1 ms to the overview's `opaque_meshes` (9.3 to
10.5 ms, means of four interleaved runs varying by 1 ms each) and nothing
measurable at street level; frame rates stayed within run-to-run noise.

## Consequences

Only what is on screen reflects: objects outside the view, behind others or
under transparent surfaces do not. Normals come from the depth buffer, so
rippled water reflects as a flat mirror, and reflections show the frame's
color before bloom and sun shafts.
