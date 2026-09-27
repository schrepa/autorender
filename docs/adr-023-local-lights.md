# Decision 023: Clustered local lights

Status: adopted for every renderer path. `LocalLights` is inserted by
default and costs nothing without `LocalLight` components.

## Context

The sun was the only light. At night, lamps, windows and neon could glow
(decision 019) but lit nothing around them, and a light on a moving vehicle
would have to be re-extracted every frame by the CPU while its mesh moves
with interpolated poses on the GPU (decision 014).

## Decision

`LocalLight` is a component: linear color, intensity (the light at one
world unit), a range where it windows to nothing, a local offset and an
optional spot cone (a local direction and inner and outer half-angles).
Light reaching a surface `d` away is `color * intensity / d²`, smoothly
windowed at the range, with the same GGX highlights as the sun. Unusable
values skip the light.

- **Where lights live.** A light on an entity that draws a mesh is extracted
  once in that mesh slot's model space; a compute pass transforms it by the
  pose the frame already has, so it follows interpolated, GPU-presented
  meshes and costs no CPU work while unchanged. Lights on other entities use
  their world pose (presented on CPU frames, authoritative on GPU frames).
  Extraction rebuilds only when a light, the settings or slot ownership
  change.
- **Clusters.** A compute pass bins each light's bounding sphere (a spot's
  tighter cone sphere) into 16×9 screen tiles by 24 exponential depth
  slices, testing each candidate cluster's view-space box, with at most 32
  lights per cluster. Lit pipelines are separate entry points
  (`fs_*_lit`), created with the first lit frame, so frames without lights
  run the unchanged shaders.
- **Settings.** `LocalLights { enabled, max_lights, intensity, distance }`
  (decision 029 makes `max_lights` an optional budget, none by default, and
  adds `lights_per_cluster`): beyond `max_lights` the lights nearest the camera win,
  chosen every frame; `intensity` scales every light (dim lamps by day);
  light fades out over the last tenth of `distance`, past which lights are
  not binned. Missing or disabled settings shade no local lights.
- **Authoring and agents.** Scene version 6 entities author `light`
  (`spot` optional, angles in degrees); the environment's `local_lights`
  section sets the settings. `LocalLight` and `LocalLights` inspect in
  those shapes, and the `environment` control changes the settings.
  `rendering` reports `local_lights`, `dropped_lights`, and, read back a few
  frames late, `lights_visible` (lights that reached a cluster) and
  `light_cluster_overflow` (entries lost to full clusters).

## Observable behavior

- Lights on moving meshes stay attached in GPU-presented frames, captures
  and previews alike.
- The Stress Lab city's street lamps carry spot lights, brighter as the
  look darkens and off by day; `effects=no-lights` turns them off.

## Measurements

None yet. In the 90,000-object city at golden hour, 4,096 lamps are
extracted (5,480 beyond the budget) and a few hundred reach clusters with no
overflow. Per-pixel cost grows with the lights per cluster; keep ranges
short and `distance` modest.

## Consequences

Lights cast no shadows. Clusters list at most 32 lights, and binning order
decides which survive an overflow; the overflow counter says when a scene
needs shorter ranges. Transparent surfaces are lit; particles are not.
