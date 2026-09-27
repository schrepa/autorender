# Decision 029: Budgets instead of caps

Status: adopted. Capacity limits are settings with no budget by default;
only the GPU's own limits bound content, and exceeding a budget drops the
least important work and reports it instead of rejecting the frame.

## Context

Prototype constants capped how much content the engine accepted: 1,024
particle emitters (the Stress Lab city has 3,174, so distant chimneys never
smoked), 4,096 local lights, 262,144 instances and transform nodes, 128
meshes and 64 textures (32 MiB each) on the GPU, 64 textures in the asset
library and 2,048-texel textures. Several were tied together through a
16-bit draw key, games could not change them, and exceeding the renderer's
rejected the frame. A high-performance engine for developers, and a stress
lab that measures its raw cost, need neither silent caps nor hard failures.

## Decision

Constants are sorted into three kinds:

- **Capacity** becomes a setting whose default sets no budget:
  `Particles::max_particles` and `LocalLights::max_lights` (`Option`, as
  scene version 6 fields), `EngineConfig::max_render_instances`,
  `EngineConfig::asset_budget` (`AssetLibrary::set_budget` at run time) and
  `EngineConfig::max_transform_nodes` (`TransformNodeLimit`). Over a budget
  the nearest emitters and lights win, extraction drops instances and
  propagation leaves extra nodes without poses, each counted (`dropped_*`,
  `overflow_transforms`). Local light clusters list
  `LocalLights::lights_per_cluster` entries (32 by default).
- **Device limits** bound what remains. The renderer requests the adapter's
  buffer and storage-binding sizes, grows buffers to the content, derives
  its instance limit from them, and drops particles that do not fit a
  binding. Textures may be 8,192 texels a side, the size every WebGPU device
  holds.
- **Validity guards** stay: they reject invalid data, not large data
  (emission scales, coordinate ranges, hierarchy depth, one emitter's
  particles within 16,777,216), as do development-protocol bounds.

The draw planner groups instances by compact 32-bit IDs of the mesh,
texture and opacity combinations in use, in the same order as before, so no
mesh or texture count bounds it; shadow selection reads each mesh group's
region start from a storage buffer.

## Observable behavior

- Every chimney in the Stress Lab city smokes (about 128,000 particles at
  90,000 objects) and every street lamp lights; `rendering` reports
  `dropped_particles` and `dropped_lights`, zero without budgets.
- The Stress Lab accepts up to 4,000,000 objects (a sanity bound on its
  URL); 500,000 build and render.
- A scene with any number of meshes and textures renders; only GPU memory
  bounds it.

## Measurements

None yet; the budgets do not change per-frame work until content exceeds
the old caps.

## Consequences

Content can now exhaust GPU memory or buffer limits; games that need
predictable frame times set budgets. Collision (100,000 nodes, 4,096 query
hits) and the 1 MiB scene source limit remain constants for now.
