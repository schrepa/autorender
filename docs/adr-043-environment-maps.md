# Decision 043: Environment maps and reflection probes

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

## Reflection probes

The sky's map holds only the haze below the horizon, so metal, glass and
mirrors reflected the city only where the screen showed it. Reflection
probes capture it:

- `ReflectionProbe { extent, blend }` on an entity (scene version 8
  `reflection_probe`) is a world axis-aligned box centered on the entity's
  world position, reaching `extent` along each axis, fading out over `blend`
  inside its faces. The `ReflectionProbes` resource (environment
  `reflection_probes`, inserted by default) sets whether probes capture, the
  `resolution` of every map face, the sky's included (a power of two from 16
  to 1,024, 128 by default), an optional `max_probes` budget and the
  `faces_per_frame` captured while probes wait (1).
- Extraction lists the usable probes nearest the camera first, by distance
  from the eye to their boxes; beyond the budget the nearest win and the rest
  count as `dropped_reflection_probes`. The renderer keeps them in more cubes
  of the array, after the sky's, as many as the device's texture array layers
  hold (`max_texture_array_layers / 6`, 42 cubes at WebGPU's 256 layers); the
  array grows by powers of two, copying what it holds, and further probes are
  dropped and counted. A probe keeps its cube while it stays among them;
  newcomers take the cubes probes leave.
- After each GPU-presented frame's drawing, the renderer captures up to
  `faces_per_frame` faces of the nearest waiting probe (never captured, or
  captured before the sky last changed), each face once a frame, and blurs a
  probe's cube like the sky's when its sixth face is drawn; from then on
  surfaces reflect it. A sky change keeps captures reflected until they are
  recaptured.
- A face draws the frame's still opaque instances without joints through a
  mirroring view (triangles wind clockwise), culled against the face and
  compacted as the frame's candidates are, and left undrawn where their
  bounds cover no texel center (decision 041); then the sky's haze. Slots are
  still when they displayed the same pose, bit for bit, for 16 frames in a
  row: a compute pass after culling counts it for every candidate, only on
  frames with probes waiting, and captures begin once it counted 16 frames,
  so moving objects never freeze into reflections. Faces are lit like the
  frame (the sun through the far shadow maps, since the near cascades follow
  the camera), with the local lights without a slot or on still slots, binned
  into the face's own clusters, and fogged, before the grade. Screen effects,
  transparency, particles and decals are left out.
- Surfaces find their probes through a grid over the ground plane, at most
  64 cells across the probes' boxes, each listing the probes reaching it,
  smallest boxes first, with their records inline. Every probe whose box holds
  the surface point reflects along the ray projected onto its box's walls
  (parallax correction), weighted by its fade, until they cover it; the sky's
  map covers the rest. Screen-space reflections replace a share where they
  hit, as before. CPU-prepared frames (captures, previews) read what earlier
  GPU frames captured.
- The Stress Lab Metro puts a probe at every street crossing, 2.5 m up, with
  boxes reaching 16 m and fading over their outer 2 m, so neighbors overlap by
  4 m; `?effects=no-probes` removes the resource.

## Observable behavior

- `memory.gpu_environment_bytes` reports the radiance cube and the maps,
  the probes' cubes included (about 1 MB each at 128 texels).
- Captures, previews and GPU frames shade alike; the maps redraw when the
  look changes (the Stress Lab's `?look=`).
- `rendering.reflection_probes` counts the probes surfaces reflect,
  `reflection_probe_faces` the faces a frame captured and
  `dropped_reflection_probes` those a budget or the device left out;
  profiles time `probe_captures`. Agents inspect `ReflectionProbes` and
  entities' `ReflectionProbe` and try settings with the `environment`
  control's `reflection_probes` section.

## Measurements

Firefox on the laptop's integrated GPU, Stress Lab Metro at 1512x792, three
interleaved runs against the sky mix: `opaque_meshes` took 12.7 ms against
12.7 in the overview and 6.7 against 6.8 at street level, and frame rates
stayed within run-to-run noise (which varied single runs by up to 3 ms).
The maps redraw only when the sky changes.

Probes, measured in one page that turned them off and on again five times
through the `environment` control, 41 probes captured in the overview: with
their captures done, `opaque_meshes` took 0.3 ms more than without them
(pairs from -0.8 to +0.7 ms, within noise) and presentation's culling the
same 0.61 ms, since stillness is counted only while captures wait (counting
every frame cost 0.24 ms). A captured face took 0.8 ms with little in it and
2.6 to 4.1 ms at the median, up to 5.9 ms at the 90th percentile, for faces
looking down streets; before faces skipped sub-texel instances, 4.5 ms at the
median. The 246 faces of 41 probes take four to eight seconds at one face a
frame, during which frame rates fell from 42 to 35 frames per second.

## Consequences

Rough surfaces reflect the sky's gradient as widely as their lobe spreads,
so bumped rough metal streaks gently instead of in hard bands; mirrors
reflect it as before, from a map of 128 texels a face. Inside probes, metal,
glass and mirrors reflect the streets, buildings and lit lamps around them
wherever the screen does not show them; elsewhere the sky's map holds the
haze's ground tint below the horizon.

Probes capture once: still objects that appear, move or change after their
probe's capture stay missing or stale until it is recaptured (after a sky
change, a new resolution, or leaving and rejoining the nearest probes), and
moving objects reflect through screen-space reflections alone. Faces draw
everything of the city they see that covers a texel center, occluded or not,
so captures cost most where streets open onto the city; culling occluded
instances within faces is the next step. Boxes project reflections onto
their walls, so they should match the spaces they stand in.
