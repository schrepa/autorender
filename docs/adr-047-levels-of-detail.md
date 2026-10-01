# Decision 047: Automatic levels of detail

Status: adopted for GPU-presented frames, on by default. The
`LevelOfDetail` resource (scene version 8 environment `level_of_detail`,
Stress Lab `?lod=off`) configures or disables it.

## Context

Ordinary meshes drew every triangle at every distance. From the Stress Lab
Metro's aerial overview most instances are a few pixels tall, so the opaque
pass rasterizes cars, lamps and planters of 60 to 250 triangles as clouds of
sub-pixel triangles. Each one still costs vertex work and triangle setup,
and the 2x2 pixel quads of tiny triangles shade several times the pixels
they cover. Virtual geometry (decision 044) already adapts dense meshes that
were built for it offline. Every other mesh, including all imported glTF
models below its threshold and every procedural mesh, had one level.

Industry practice is a chain of discrete levels per mesh, each simplified
from the last. Unreal's automatic LODs and meshoptimizer's recommended chains
are two examples. Each level's geometric error selects it at runtime, either
from a screen-size threshold or from the error projected into pixels.
meshoptimizer 1.3's simplifier (quadric error with attribute metrics, seam
protection and component pruning) is the most widely used implementation.

## Decision

### Chains

`engine_assets::simplify` is a pure-Rust port of meshoptimizer 1.3's
simplifier: `simplify` with its options, attribute weights and vertex flags,
`simplify_scale` and `protect_attribute_seams`, ported from `clusterlod.h`.
It is Rust because the Wasm build has no C toolchain. `tools/vgbuild` checks
it against the C++ library: indices and error bits match exactly over 1,320
structured cases, 33 cases on a 100,000-triangle mesh and 4,000 random
meshes. It takes about as long as the C++ natively and in Wasm (a
101,760-triangle sphere reduced by half with normals and texture
coordinates: 33.7 ms C++, 31.5 ms Rust, 32.8 ms Wasm).

`engine_assets::mesh_lods(mesh, max_levels)` builds the chain when the
renderer uploads a mesh. Meshes with joints, virtual geometry or fewer than
`LOD_MIN_TRIANGLES` (64) triangles get none.

- Each level simplifies the previous one toward half its triangles, using
  the mesh's normals and texture coordinates as attributes. Texture
  coordinates are weighted per unit of their own range.
- Texture seams are protected. Normal seams, which split every edge of
  faceted meshes, may be crossed.
- The options are meshoptimizer's recommended chain: absolute errors,
  permissive collapses, clamped errors and pruning of small disconnected
  parts.
- There is no error limit. A level is drawn only where its error projects
  within the pixel budget.
- The chain stops at a level that keeps more than 85% of the last one's
  triangles or has fewer than 16.
- Each level's error is at least 1.5 times the last one's, so errors grow
  strictly.

Levels are index lists over the mesh's own vertices. The arena (decision
046) stores them after the mesh's indices in the same range, so a level
costs indices only. Draw groups have at most `LOD_LEVELS` (3) levels: the
mesh and two simplifications.

### Selection

GPU-presented frames that cull without an instance budget pick each
frustum-visible instance's level in the culling pass (`lod_level` in
`presentation.wgsl`).

- Level `l` draws where `(error_l * s * K + r * s)^2 <= |c - eye|^2`:
  - `s` is the instance's largest axis scale;
  - `r` is its mesh's bounding radius and `c` its posed bounding center;
  - `K` is the projection's pixels per world unit at unit distance divided
    by `pixel_error`.
- It measures distance from the eye, not depth, so turning the camera
  never changes a level.
- Orthographic views compare `error_l * s * K` with 1.

The level takes two flag bits beside the visibility flags.

### Compaction and draws

The compaction passes (decision 014) count, scan and scatter four buckets
at once as `vec4` lanes: levels 0 to 2 and impostors (decision 048).

- Each level has its own plane of visible IDs and its own indirect
  arguments. Level `l` of batch `b` draws from `draws[l * batches + b]` with
  the level's index range.
- With `indirect-first-instance`, a level's first instance adds `l` planes.
  Without it, the draw binds the level's region of the plane.
- Level 0 keeps the previous layout, so virtual geometry and occlusion
  culling read it unchanged.
- An instance budget keeps its meaning by disabling levels on frames that
  have one.

The two occlusion phases (decision 040) each compact their own levels.

### Shadows

Each shadow map picks one level per draw group: the coarsest whose error is
within half a texel of that map. The kept and moving maps use their far
texel, and each cascade uses its own. The moving map, whose casters are drawn
every frame, allows a whole texel: its outlines may then move by one texel.
The level changes only when a map's texel does, and the argument template
is rewritten then. Every instance in the group casts at that level.

### Everything else draws meshes

CPU-presented frames, captures and previews without `gpu`, reflection probe
faces, transparent groups, skinned meshes and virtual geometry draw level 0.
CPU frames stay the exact reference (decision 014). GPU-presented frames
differ from them only by simplifications within `pixel_error`.

## Observable behavior

- `LevelOfDetail { enabled: true, pixel_error: 1.0 }` is inserted by
  default. `pixel_error` ranges from 0.25 to 64 pixels of viewport height.
  Removing the resource or disabling it draws every mesh at level 0.
- Scene version 8 environments author `level_of_detail`. The paused
  `environment` control changes it temporarily, and inspection reports it.
  `BrowserApp` hosts and the Stress Lab's `?lod=off` turn it off.
- `rendering.lod0_instances`, `lod1_instances` and `lod2_instances` count
  the instances drawn at each level, read back a few frames late.
  `impostor_instances` counts those drawn as impostors
  ([decision 048](adr-048-impostors.md)). Profiles count
  `render.lod0_instances` to `render.lod2_instances`.
- `web/scripts/lod-smoke.mjs` checks that:
  - the settings are discoverable;
  - coarser levels draw from the Metro overview, within 1% of the exact
    frame's pixels;
  - disabled frames match exactly;
  - a larger pixel error draws more coarsely;
  - invalid settings are refused.

Measurements:
- Stress Lab Metro, 90,000 objects, Chrome, 1200x900 canvas: the overview
  capture differs from the exact frame in 5,425 pixels (0.5%). A 4x crop
  shows no visible difference, and the close view is identical.
- The overview draws 47,595 instances at level 0, 9,512 at level 1 and 1,504
  at level 2. Its camera draws submit 3.25 million triangles at full detail,
  1.84 million with levels and 1.13 million with impostors too
  (`rendering.triangles`, 1920x1080 window); the CPU path, which culls by
  frustum alone, submits 4.98 million.
- Firefox on the laptop's integrated GPU, one build without impostors
  (decision 048), three alternating runs each:
  - opaque pass 9.60 ms off, 7.45 ms on;
  - throughput 49.0 to 52.8 fps off (mean 50.7), 55.9 to 57.4 on (mean
    56.6).
  - An earlier pair of runs, while the machine was also building, measured
    10.33 ms and 9.12 ms (43.4 and 45.8 fps).
- The moving map's whole-texel levels (Firefox, integrated GPU, Metro
  overview, 2262x968 canvas, two runs each, 30 September 2026):
  - `shadow_moving` fell from 1.54 to 1.08 ms.
  - Clearing the map without drawing costs 0.19 ms.
  - Drawing every moving caster at its coarsest level costs 0.95 ms. Choosing
    levels per instance by distance could therefore save at most 0.14 ms
    more, so it was not built.
  - Chrome captures differ in 0.1% of the overview's pixels, in single far
    shadow texels, and not at all in the close view.

## Consequences

- Chains are built on the main thread when a mesh first uploads. A 200-triangle
  prop takes microseconds and a 100,000-triangle mesh about 50 ms, once.
- Meshes carry up to about 75% more indices.
- Metro's props are mostly closed boxes of a few triangles each. Nothing
  collapses in them without deleting a whole part, so their levels come
  from pruning. A 208-triangle car goes straight to 36 triangles with an
  error of 0.3 m, which is drawn only where 0.3 m is under a pixel. Props
  under 64 triangles get no levels at all.
- The instances that remain costly from afar are many small instances, not
  their triangles. Impostors ([decision 048](adr-048-impostors.md)) are the
  coarsest level.

## Alternatives and evidence

- **Per-level error bounds** (1% and 5% of the mesh's extent, tried 29
  September 2026). Boxy low-poly props cannot simplify within them, so no
  Metro model got a single level.
- **Pruning only in the last level.** The first level then kept over 85%
  of nearly every model's triangles, and only one Metro model got a level.
- **Hysteresis between levels.** Not added: at a one-pixel error a level
  switch changes no more than a pixel.
- **Screen-size thresholds.** Rejected in favor of projected error. A
  threshold would need tuning per mesh, whereas the error comes from the
  simplifier.
