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
`max_roughness` and through fog. At half resolution, from every reflective
pixel, a ray along the mirror direction of its depth normal takes 24 steps
(closer near the surface, offset by fixed noise) up to `distance`; the first
step behind a surface is refined by bisection and kept if the ray passed
close behind it. The composite, or what the temporal resolve shows,
replaces the pixel's reflectivity-weighted share with the reflected scene
color, faded toward the screen edges and the ray's end. Misses keep the sky
reflection already shaded.

## Observable behavior

- Captures, previews and GPU frames match; the pass reports
  `rendering.reflections`.
- The Stress Lab opts in; `effects=no-reflections` removes the resource.

## Measurements

None yet: only reflective pixels trace, at half resolution.

## Consequences

Only what is on screen reflects: objects outside the view, behind others or
under transparent surfaces do not. Normals come from the depth buffer, so
rippled water reflects as a flat mirror, and reflections show the frame's
color before bloom and sun shafts.
