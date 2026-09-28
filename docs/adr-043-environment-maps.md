# Decision 043: Environment maps

Status: adopted for every renderer path.

## Context

Surfaces reflected the analytic sky (decision 018) in their mirror
direction, blurred toward its hemisphere average by a linear mix with
roughness (decision 019). At roughness 0.3 the mirror direction kept 70% of
the reflection, so the brushed metal of the Metro's tanks swept each groove
of its normal map across the horizon's gradient and streaked in hard orange
and violet bands, while screen-space reflections (decision 025), blurred by
the same roughness lobe, drew that metal soft wherever they found something
on screen. The reflection a surface showed depended on which of the two
drew it.

## Decision

The renderer keeps an array of environment maps: cubes of 128 texels a
face in `Rgba16Float`, with eight levels. The first holds the sky. Whenever
the atmosphere's sun, sky or ground colors change:

- the sky's haze (without the sun's disc, whose highlights surfaces shade
  themselves) is drawn into the faces of a radiance cube, and its levels
  averaged down;
- each level of the environment map is blurred from the radiance cube by
  the GGX lobe of roughness `level / 7`, seen along the texel's direction
  (the split sum's assumption): 64 Hammersley samples per texel, each read
  from the radiance level whose texels match its share of the lobe (Karis's
  filtered importance sampling). The first level copies.

Surfaces read their reflection from the environment map's level of their
roughness, along the lobe's dominant direction, which leans toward the
normal as the surface roughens (Lagarde and de Rousiers), weighted by the
environment BRDF as before. Screen-space reflections replace a share of it
where they hit, so their misses show the same blurred environment.

The maps bind in the main passes' first group (a cube array at binding 10
and its sampler at 11): lit pipelines use all four groups the device allows.
The passes drawing the maps bind only their own groups, since a pass cannot
read a texture it draws into.

## Observable behavior

- `memory.gpu_environment_bytes` reports the radiance cube and the maps.
- Captures, previews and GPU frames shade alike; the maps redraw when the
  look changes (the Stress Lab's `?look=`).

## Measurements

Firefox on the laptop's integrated GPU, Stress Lab Metro at 1512x792, three
interleaved runs against the sky mix: `opaque_meshes` took 12.7 ms against
12.7 in the overview and 6.7 against 6.8 at street level, and frame rates
stayed within run-to-run noise (which varied single runs by up to 3 ms).
The maps redraw only when the sky changes.

## Consequences

Rough surfaces reflect the sky's gradient as widely as their lobe spreads,
so bumped rough metal streaks gently instead of in hard bands; mirrors
reflect it as before, from a map of 128 texels a face. Below the horizon the
map holds the haze's ground tint: only what the screen shows reflects the
city.
