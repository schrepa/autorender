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
  change, and retained frames name the rebuild (`RenderFrame::lights_revision`)
  so renderers restage and upload light inputs only then; owned snapshots
  compare their inputs' bytes.
- **Clusters.** A compute pass bins each light's bounding sphere (a spot's
  tighter cone sphere) into 16×9 screen tiles by 24 exponential depth
  slices, testing each candidate cluster's view-space box. Each workgroup
  places eight lights and its threads share their candidate clusters, so a
  light near the eye covering thousands of clusters takes a whole
  workgroup instead of one thread. Binning counts each cluster's lights,
  one workgroup scans the counts into list ranges,
  and a second pass fills the lists, so every list is exactly as long as
  the lights reaching its cluster. List storage grows to the entries a
  frame read back needed (a first guess of 32 per light before that). Lit
  pipelines are separate entry points (`fs_*_lit`), created with the first
  lit frame, so frames without lights run the unchanged shaders.
- **Settings.** `LocalLights { enabled, max_lights, lights_per_cluster,
  intensity, distance }` (decision 029 makes `max_lights` and
  `lights_per_cluster` optional budgets, none by default): beyond
  `max_lights` the lights nearest the camera win, chosen every frame; a
  cluster lists at most `lights_per_cluster` lights, the first binned;
  `intensity` scales every light (dim lamps by day);
  light fades out over the last tenth of `distance`, past which lights are
  not binned. Missing or disabled settings shade no local lights.
- **Authoring and agents.** Scene version 6 entities author `light`
  (`spot` optional, angles in degrees); the environment's `local_lights`
  section sets the settings. `LocalLight` and `LocalLights` inspect in
  those shapes, and the `environment` control changes the settings.
  `rendering` reports `local_lights`, `dropped_lights`, and, read back a few
  frames late, `lights_visible` (lights that reached a cluster) and
  `light_cluster_overflow` (entries lost to the per-cluster budget, or to
  list storage in the frames before it grows).

## Observable behavior

- Lights on moving meshes stay attached in GPU-presented frames, captures
  and previews alike.
- The Stress Lab city's street lamps carry spot lights, brighter as the
  look darkens and off by day; `effects=no-lights` turns them off.

## Measurements

No timings yet. The 90,000-object city at night extracts 9,576 lamps; at
street level 169 reach clusters. Seen from above with `distance` 2,000,
2,581 reach clusters and every entry lists (overflow 0), where a budget of
32, the former fixed list length, loses 1,633.

## Consequences

Lights cast no shadows. Binning visits each light's clusters twice (count
and fill) plus a one-workgroup scan of the 3,456 clusters. On the Stress
Lab Metro street-level view (947 of 11,418 lights visible, Firefox on an
integrated GPU) sharing clusters within workgroups cut binning from 0.83
to 0.34 ms; the aerial overview went from 0.14 to 0.20 ms. Per-pixel cost
grows with the lights per cluster: keep ranges short and `distance` modest,
or set `lights_per_cluster`. Transparent surfaces are lit; particles are
not.
