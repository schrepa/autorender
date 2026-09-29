# Stress Lab

Open `/stress.html` on the running server. The page starts with 90,000 textured
towers and 32 mesh batches. Select a preset to replace the workload, or adjust
individual controls and Apply (up to 4,000,000 objects, a sanity bound of the
page; the engine sets no budget, decision 029). Failed configuration preserves the active
world.

| Preset | What it exercises |
| --- | --- |
| Maximum visual | 90,000 static objects, 32 meshes and textures; no collision plugin or motion systems |
| Wild city | 90,000 objects of 52 shapes in a procedural city; 10% are traffic and machines, with hierarchies and interpolation |
| Mixed | 10,000 objects, moving hierarchies/interpolation, 1,000 colliders and 32 queries per tick |
| Transforms | 50,000 moving objects with parent hierarchies and interpolation |
| Queries | 10,000 colliders, moving bounds and 256 ray/overlap/sweep queries per fixed tick |
| Dense sensors | 2,500 sensor colliders in tight clusters; pair-budget exhaustion is deliberately observable |
| Dense geometry | 2,000 boulders and columns of 330,000 to 590,000 triangles each on terrain tiles, drawn as virtual geometry |

The `scene` configuration field selects the world: `field` (the tower
field above, the default), `city`, `metro` or `dense`. Configurations and links
without `scene` build the tower field, unchanged.

## Wild city

The city is procedural Rust in `examples/stress-lab/src/city.rs` and
`shapes.rs`, deterministic per seed. Every mesh fits the unit cube, so an
object's scale is its size, and textures are generated at startup (façades of
30 floors, roads, water, grass, foliage, metal, vehicles). The same settings
apply with city meanings:

- **Objects** are exactly the renderables. Static ones fill 20-unit blocks
  between 4-unit streets outward from the center: a skyscraper core and a few
  secondary downtowns, midtown, low-rise streets, industrial outskirts,
  parks, plazas, construction sites, stadiums, a river with bridges and an
  elevated rail ring around the core.
- **Moving objects** are the traffic: cars and buses on block loops, trains
  on the rail ring, boats, drones carrying loads, helicopters with spinning
  rotors, crane jibs with their loads, Ferris wheels, rooftop beacons and
  façade elevators. At 100% the city is traffic only.
- **Parent hierarchies** make each block's parts children of a transform-only
  block node, and moving assemblies (crane jib and load, wheel and cabins,
  drone and cargo, helicopter and rotor) children of moving pivots. Without
  them every part is a root whose motion reproduces the same poses; tests
  compare both layouts. Block and pivot nodes stay within the
  transform-node limit, falling back to roots.
- **Mesh batches** keep 64 (all 52 shapes), 32, 8 or 1 shape; other shapes
  fall back to a simpler one. Draw groups are the mesh and texture pairs in
  use: 91 at 64, 36 at 8.
- **Colliders** spread evenly over the objects; dense clusters place them on
  crates at the tower field's cluster positions.

The inside view looks across the river at the core from street level.
The Lighting control picks an atmosphere ([decision 018](adr-018-atmosphere-and-stylized-lighting.md),
graded per [decision 027](adr-027-stylized-grading.md)):
the scene's own (night over the tower field, golden hour over the city),
golden hour, dusk, night or day. `look=golden|dusk|night|day` in the URL sets
it at startup; it survives workload changes, and recordings and profiles
report `look`.
Materials use surfaces ([decision 019](adr-019-surface-materials-and-reflections.md)):
the city's glass towers, metal, vehicles and water reflect the sky, the river
ripples, and lit windows, neon, beacons and lamp heads glow through their
textures' emission masks; the tower field varies roughness and metalness and
lights every fourth tower's stripes. Each scene bounds its shadows
([decision 020](adr-020-sun-shadows.md)) at 4096 texels. The city's street lamps
carry spot lights ([decision 023](adr-023-local-lights.md)), brighter as the
look darkens and off by day; chimneys smoke and fountains spray particles, and
glass domes are transparent ([decision 026](adr-026-transparency-and-particles.md));
both scenes opt into sun shafts ([decision 024](adr-024-sun-shafts.md)) and
screen-space reflections ([decision 025](adr-025-screen-space-reflections.md)).
The Effects control turns shadows, ambient occlusion and bloom
([decision 021](adr-021-ambient-occlusion-and-bloom.md)), local lights, sun
shafts, reflections, particles, refraction or reflection probes
([decision 043](adr-043-environment-maps.md)) off, or all of them; `effects=none` or a
comma-separated list such as `effects=no-shadows,no-shafts` sets it at
startup, it survives workload changes, and recordings and profiles report
`effects`. The measurements below predate these effects.
The city averages about 60 vertices and 29 triangles per object, against 24
and 12 for the tower field, so the opaque pass does most of the added work.
In Firefox 156 on the Intel iGPU (canvas 2262×987, 10% moving, no collision,
two page loads each):

| Workload | Frames per second | Opaque pass GPU (ms) | Main thread per frame (ms, median) |
| --- | --- | --- | --- |
| Tower field, 90,000 objects, 32 meshes | 118–120 | 1.95–2.0 | 1.0–1.2 |
| City, 90,000 objects, 52 shapes | 101–104 | 4.2–4.75 | 1.3–1.4 |
| City, 250,000 objects, 52 shapes | 69–71 | 8.4–9.8 | 2.5–3.3 |

With 8 shapes (41 vertices per object) the 90,000-object city's opaque pass
measured 3.1 ms. Temporal antialiasing adds its resolve (1.3 ms at this size).

## Dense geometry

The dense scene (`scene=dense`, `examples/stress-lab/src/dense.rs`) draws
virtual geometry ([decision 044](adr-044-virtual-geometry.md)): boulders of
327,680 triangles, fluted columns of 591,360 and terrain tiles of 524,288, as
cluster hierarchies the GPU selects from each frame. Build its files once
with `cargo run -p engine_vgbuild --release --bin dense_assets`, which writes
them to `web/public/stress/dense/` (generated, not committed); the page
fetches each file's prefix (its metadata and coarsest levels, about 2 MB in
all) when the scene is first chosen, and the renderer streams the other
pages as views need them. **Objects** are the boulders
and columns (one in five a column), on as many 64-unit tiles as give each
about 36 square units; **moving objects** bob and turn in place. The other
settings do not apply. The Effects control and looks work as elsewhere, and
virtual meshes cast their clusters' shadows. Frames report the clusters and
triangles the camera and the shadow maps drew (`virtual_clusters`,
`virtual_triangles`, `virtual_shadow_clusters`, `virtual_shadow_triangles`),
the camera's clusters occlusion culling left undrawn
(`virtual_occluded_clusters`, with `?occlusion=off` to compare), any
entries dropped for want of room (`dropped_virtual`), and the pages resident,
loading, wanted and lacking, and finding no room (`virtual_resident_pages`,
`virtual_loading_pages`, `virtual_missing_pages`, `dropped_virtual_pages`).
Clusters shade through the visibility buffer
([decision 045](adr-045-visibility-buffer.md), `visibility_buffer`);
`?visibility=off` shades them as they draw, for comparison.

## Measurement

Overview, inside-field and look-away cameras isolate visibility changes. Culling
and resolution controls isolate draw submission and pixel costs; mesh variety
changes actual draw batches. Resolution remains bounded by the shared host's
2560�1440 drawing-buffer budget. These settings do not imply GPU timestamps.

Click Record in the workload panel to capture while the scene stays visible,
then Export JSON. Camera input and workload controls lock during recording.
Pause or changing visibility aborts with an incomplete report. Copy workload link
preserves the configuration and seed and opens a fresh overview view. The
source of this scene is procedural Rust plus the explicit JSON configuration,
not the 10,000-entity authored-scene file format.

Agents can launch `cargo xtask agent watch --game stress`, discover `StressConfig`
and `StressMetrics`, and use the standard inspect/step/benchmark interfaces. A
`config` JSON query parameter selects a repeatable workload at startup. Ordinary
frames use GPU presentation ([decision 014](adr-014-gpu-presentation-and-culling.md));
`presentation=cpu` keeps the exact CPU presentation path for same-build
comparisons, and recordings and profiles report the selected `presentation`. Those
frames also pipeline fixed ticks ([decision 015](adr-015-pipelined-fixed-ticks.md));
`pipeline=off` runs each tick when due, and recordings report the setting.
`taa=on` temporally antialiases GPU-presented frames
([decision 017](adr-017-temporal-antialiasing.md)); recordings and profiles
report `taa`. `occlusion=off` draws every frustum-visible object without the
occlusion test ([decision 040](adr-040-occlusion-culling.md)), and
`subpixel=off` also every instance covering no pixel center
([decision 041](adr-041-subpixel-culling.md)), and `lod=off` draws every
instance as its mesh instead of at its level of detail
([decision 047](adr-047-levels-of-detail.md)), and `impostors=off` draws no
impostors ([decision 048](adr-048-impostors.md)), for same-build comparisons;
recordings report `occlusion`, `subpixel`, `lod` and `impostors`. Spatial
queries require the collision-enabled presets. No automated suite starts when
opening the page. Runtime code lives in `examples/stress-lab`, independent of
engine implementations.

The Stress Lab has an opt-in production recording helper. Recording observes
the running workload and exports JSON; it does not enable the development
protocol, pause gameplay, execute extra simulation steps or change the scene.

Keep the same seed, configuration, camera, culling setting and drawing resolution
when comparing recordings. The page includes these settings in the report's
metadata. The page records the startup Wasm URL and SHA-256 when available, plus
the managed build ID; absent fingerprints are explicit nulls. Never rebuild a
mutable development server during a run. Browser identity, viewport, device pixel ratio and available browser
hardware hints are captured once at recording start. An existing runtime's
adapter description may accompany the metadata; the helper never requests a
second WebGPU adapter or device. Browser hints do not establish physical GPU use.

A default recording discards 30 warm-up frames, then measures ten seconds of
animation-frame intervals. Its sample count depends on the browser's actual
cadence: a slow workload can produce a complete recording with fewer frames.
CPU work for each frame is paired with the **following** animation-frame
interval, including the last measured frame. Reports preserve the raw timing
samples and min/median/p95/p99/max distributions. Percentiles use linear
interpolation at `(sample_count - 1) × fraction`.

CPU total is simulation plus extraction plus render preparation/submission.
Additional phase timers, such as collision work inside simulation, are subsets;
adding them again would double-count their cost. CPU stage timers exclude
recording/stat collection, while frame intervals include host work and browser
pacing. **GPU execution time is not measured.** These recordings also do not
measure JS/Wasm/process heaps or total driver/GPU memory.

Per-frame numeric counters retain their first/last values and measured minima
and maxima, exposing transient missing assets, invalid transforms, overflows and
other diagnostics even when the last frame has recovered. Boolean completeness
counters use 0/1 minima and maxima. Stress-specific JSON
snapshots are retained at most four times per second. Dropped simulation time is
the change in the cumulative counter over the measured frames; with zero warm-up,
the first observed frame has no preceding counter baseline.

Configuration, camera, resize, visibility and pause changes must abort an active
recording through the page's lifecycle hooks. An interrupted report remains
exportable with `complete: false` and its reason. A completed report means the
time window finished; inspect diagnostic counters before treating a workload as
healthy or comparable. There is a 30-second wall-time deadline including warm-up
and a cap of 1,800 measured samples. Reaching either limit before ten measured
seconds produces an incomplete report. Metadata is capped at 16 KiB; individual
diagnostic snapshots at 8 KiB, with at most 121 snapshots per recording.

Close unrelated browser workloads and avoid builds during comparisons. Repeat
matched runs and retain each JSON artifact: browser scheduling, power state and
background work can change results. Use the separate [benchmark runner](benchmarking.md)
for automated fresh-page repetitions with compatibility checks.

The page owns UI, export and runtime lifecycle. `web/src/stress-recording.ts`
provides `StressRecorder.start(metadata, options)`, `sample(frame)`,
`diagnostic(timestamp, snapshot)` and `abort(reason)`. `active` allows the host to
read per-frame Wasm stats only during recording; ordinary rendering retains its
existing low-frequency HUD polling. Stats bindings remain owned by the host and
must still be released after copying their fields.
