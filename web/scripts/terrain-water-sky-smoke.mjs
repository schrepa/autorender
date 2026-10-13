/**
 * The terrain fixture's water and sky: rivers, lakes, the sea, clouds and sun.
 *
 * The hydrology bake finds channels and falls; lakes and the sea stand
 * where queries find them, a buoy rides the sea, and the sea's water veils
 * what lies under it; an interior darkens its ground; clouds, the physical
 * sky and the time of day draw, read back and refuse settings out of range;
 * a low sun casts the horizon map's long shadows, thick air hazes the
 * terrain, and running frames capture the reflection probe. The fixture:
 * terrain-fixture.mjs.
 */
// @ts-check
import assert from "node:assert/strict";
import { differentPixels } from "./harness.mjs";
import { fast, openTerrain, overview, skyColor } from "./terrain-fixture.mjs";

const { page, errors, command, close, settle, capture, inspectGround } = await openTerrain("terrain-water-sky");
const checks = [];

try {
  // The hydrology bake read every sample of the two-meter
  // tiles, found channels, and queries report what it found where they ask.
  const water = (await inspectGround()).hydrology;
  assert.equal(water.baking, false, "the bake is done once tiles load");
  assert.equal(water.samples, 513 * 513, `the bake reads its level: ${JSON.stringify(water)}`);
  assert.ok(water.channel_samples > 0 && water.max_area > 20000, `water gathers into channels: ${JSON.stringify(water)}`);
  const [watered] = (await command("terrain_height", { points: [[-300, -300]] })).result.samples;
  assert.ok(watered.water && watered.water.flow > 0 && Number.isFinite(watered.water.wetness), `queries report water: ${JSON.stringify(watered)}`);
  // Waterfalls are where its rivers drop steeply; the
  // engine spawns what draws each one.
  assert.equal((await command("inspect", { component: "Terrains" })).result.value.falls, water.falls, `falls spawn as found: ${JSON.stringify(water)}`);
  checks.push({ check: "hydrology", samples: water.samples, channels: water.channel_samples, lakes: water.lake_samples, max_area: water.max_area, falls: water.falls });
  // The checks of water, the sea, interiors and the sky that follow ran
  // with the terrain control's materials view (the terrain smoke checks
  // the view itself) when one smoke held them all, and keep it.
  await command("terrain", { view: "materials" });
  // Water stands in the bake's lakes: queries find its
  // surface and depth, the overview draws chunks of it, it shades pixels
  // and the section refuses water clearer than its range.
  const grid = [];
  for (let z = -480; z <= 480; z += 40) for (let x = -480; x <= 480; x += 40) grid.push([x, z]);
  const lakes = [];
  for (let i = 0; i < grid.length; i += 100) {
    const samples = (await command("terrain_height", { points: grid.slice(i, i + 100) })).result.samples;
    lakes.push(...samples.filter(sample => sample?.standing_water));
  }
  assert.ok(lakes.length > 0 && lakes.every(sample => sample.standing_water.depth > 0
    && Math.abs(sample.standing_water.surface - sample.standing_water.depth - sample.height) < 1e-2),
    `water stands in lakes: ${lakes.length}`);
  const waterSettings = (await command("inspect", { component: "Water" })).result.value;
  await command("camera", overview);
  const wet = (await command("environment", { water: waterSettings })).result.rendering;
  assert.ok(wet.water.chunks > 0, `water draws: ${wet.water.chunks}`);
  const withWater = await capture("water");
  await command("environment", { water: { ...waterSettings, enabled: false } });
  const withoutWater = await capture("no-water");
  const waterPixels = await differentPixels(page, [withWater, withoutWater]);
  assert.ok(waterPixels.changed > 1000, `water shades ${waterPixels.changed} pixels`);
  const murky = await command("environment", { water: { ...waterSettings, clarity: 0 } }, false);
  assert.equal(murky.error.path, "$.water.clarity");
  await command("environment", { water: waterSettings });
  checks.push({ check: "water", lake_samples: lakes.length, chunks: wet.water.chunks, pixels: waterPixels.changed });
  // The sea stands at its level wherever the ground lies
  // lower: queries sum the waves frames draw, which move with time and stay
  // within their height, its sectors draw, and a camera just over the
  // surface the CPU finds sees the sea where one just under it sees the
  // ground below.
  {
    const still = (await command("inspect", { component: "Ocean" })).result.value;
    assert.equal(still.enabled, false, "the sea is off by default");
    const sea = { ...still, enabled: true, level: 40 };
    await command("environment", { ocean: sea });
    const shore = [];
    for (let z = -440; z <= 440; z += 80) for (let x = -440; x <= 440; x += 80) shore.push([x, z]);
    const tide = (await command("water_query", { points: shore, seconds: 12.5 })).result;
    assert.ok(tide.level === 40 && tide.amplitude > 0.1 && tide.waves.length === sea.waves, `the sea's waves: ${JSON.stringify(tide).slice(0, 400)}`);
    assert.ok(tide.samples.every(sample => Math.abs(sample.ocean.height - 40) <= tide.amplitude + 1e-2
      && Math.abs(Math.hypot(...sample.ocean.normal) - 1) < 1e-3 && sample.ocean.normal[1] > 0.5),
      `surfaces stay within the waves' height: ${JSON.stringify(tide.samples.slice(0, 3))}`);
    const later = (await command("water_query", { points: shore, seconds: 13.5 })).result;
    assert.ok(later.samples.some((sample, index) => Math.abs(sample.ocean.height - tide.samples[index].ocean.height) > 1e-2), "waves move");
    const [depth, deepest] = tide.samples.map((sample, index) => [sample.depth ?? -1, shore[index]]).sort((a, b) => b[0] - a[0])[0];
    assert.ok(depth > 8, `the sea stands deep somewhere: ${depth}`);
    const surface = (await command("water_query", { points: [deepest] })).result.samples[0].ocean.height;
    const down = offset => ({ position: [deepest[0], surface + offset, deepest[1]], target: [deepest[0], surface - 50, deepest[1] + 0.01], vertical_fov_radians: 0.2, near: 0.05, far: 2000 });
    await command("camera", down(0.3));
    const afloat = (await command("render")).result.rendering;
    assert.ok(afloat.ocean.sectors > 0 && afloat.ocean.triangles > 0, `the sea draws: ${afloat.ocean.sectors} sectors`);
    const above = await capture("sea-above");
    await command("camera", down(-0.3));
    const below = await capture("sea-below");
    const crossing = await differentPixels(page, [above, below], { tolerance: 24 });
    // Rivers carve their mouths deep under the sea, which
    // darkens it from above, and the sky's state (how far its clouds have
    // marched) sets how far the two views differ: between a quarter and half
    // of the view changes across the surface, where a surface drawn off the
    // queried one leaves both cameras on one side and the views alike.
    assert.ok(crossing.changed > 0.15 * crossing.total, `the drawn surface lies where queries find it: ${crossing.changed} of ${crossing.total} pixels change across it`);
    // Floating entities ride the sea each fixed tick at the
    // tick's time, which frames present: the buoy stands on the surface
    // queries find there.
    await command("step", { ticks: 1 });
    const buoy = (await command("inspect", { entity: "buoy", component: "Transform" })).result.entity.components.Transform;
    const floating = (await command("inspect", { entity: "buoy", component: "Floating" })).result.entity.components.Floating;
    assert.deepEqual(floating, { draft: 0.25, tilt: 1 }, `the buoy inspects as authored: ${JSON.stringify(floating)}`);
    const swell = (await command("water_query", { points: [[400, -400]] })).result.samples[0].ocean;
    assert.ok(Math.abs(buoy.translation[1] - (swell.height - 0.25)) < 2e-3, `the buoy floats: ${JSON.stringify(buoy)} on ${JSON.stringify(swell)}`);
    // Under the sea its water veils everything: a camera a
    // meter under the surface sees nothing like one a meter over it.
    const level = (await command("water_query", { points: [deepest] })).result.samples[0].ocean.height;
    const across = offset => ({ position: [deepest[0], level + offset, deepest[1]], target: [deepest[0] + 40, level + offset, deepest[1] + 40], vertical_fov_radians: 0.9, near: 0.05, far: 2000 });
    await command("camera", across(1));
    const dry = await capture("sea-afloat");
    await command("camera", across(-1));
    const diving = await capture("sea-diving");
    const veiled = await differentPixels(page, [dry, diving], { tolerance: 24 });
    assert.ok(veiled.changed > 0.5 * veiled.total, `under the sea, its water veils the view: ${veiled.changed} of ${veiled.total} pixels change`);
    const sparse = await command("environment", { ocean: { ...sea, waves: 0 } }, false);
    assert.equal(sparse.error.path, "$.ocean.waves");
    await command("environment", { ocean: still });
    await command("camera", overview);
    checks.push({ check: "ocean", sectors: afloat.ocean.sectors, triangles: afloat.ocean.triangles, amplitude: tide.amplitude, depth, crossing: crossing.changed, buoy: buoy.translation[1], veiled: veiled.changed });
  }
  // Inside an interior's box the ground takes a tenth of the
  // sky and sun: looking straight down at the box's edge, the half inside
  // is far darker than the half outside.
  {
    const [edge] = (await command("terrain_height", { points: [[-280, 300]] })).result.samples;
    await command("camera", { position: [-280, edge.height + 60, 300], target: [-280, edge.height, 300.01], vertical_fov_radians: 0.6, near: 0.5, far: 2000 });
    const view = await capture("interior-edge");
    const [first, second] = await page.evaluate(async png => {
      const image = new Image();
      image.src = `data:image/png;base64,${png}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = image.width; canvas.height = image.height;
      const context = /** @type {CanvasRenderingContext2D} */ (canvas.getContext("2d"));
      context.drawImage(image, 0, 0);
      const { data, width, height } = context.getImageData(0, 0, image.width, image.height);
      const sums = [0, 0], counts = [0, 0];
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          // Leave a band either side of the edge.
          if (Math.abs(x - width / 2) < width * 0.1) continue;
          const offset = (y * width + x) * 4, half = x < width / 2 ? 0 : 1;
          sums[half] += 0.2126 * data[offset] + 0.7152 * data[offset + 1] + 0.0722 * data[offset + 2];
          counts[half]++;
        }
      }
      return [sums[0] / counts[0], sums[1] / counts[1]];
    }, view);
    const [dark, light] = first < second ? [first, second] : [second, first];
    assert.ok(dark * 2 < light, `the interior darkens its ground: ${dark.toFixed(1)} against ${light.toFixed(1)}`);
    await command("camera", overview);
    checks.push({ check: "interior", inside: dark, outside: light });
  }
  // Clouds are opt-in: the fixture has none until the
  // environment control adds a layer; exact frames march the whole dome,
  // which the sky shows; the view marches them too, lying
  // over the ground from above the layer; no coverage marches nothing, and
  // settings out of range are refused.
  {
    const up = { position: [0, 60, 0], target: [300, 400, 200], vertical_fov_radians: 1.0, near: 0.5, far: 4000 };
    await command("camera", up);
    const clear = await capture("sky-clear");
    const layer = { enabled: true, coverage: 0.6, base: 600, thickness: 900, density: 1, wind: [8, 3], scale: 2000, seed: 3, shadows: 0.75, resolution: 1024, slices: 16 };
    const clouded = (await command("environment", { clouds: layer })).result.rendering;
    assert.equal(clouded.clouds.rows, 256, `exact frames march the whole dome: ${clouded.clouds.rows}`);
    const view = (await command("inspect", { component: "Clouds" })).result.value;
    assert.ok(Math.abs(view.coverage - 0.6) < 1e-6, `coverage reads back: ${view.coverage}`);
    const cloudy = await capture("sky-clouds");
    const cloudPixels = await differentPixels(page, [clear, cloudy]);
    assert.ok(cloudPixels.changed > 20000, `clouds show in the sky: ${cloudPixels.changed} pixels`);
    // Their shadows fall on the ground the sun lights.
    await command("camera", { position: [0, 400, 1], target: [0, 30, 0], vertical_fov_radians: 1.0, near: 0.5, far: 4000 });
    const shaded = await capture("cloud-shadows");
    await command("environment", { clouds: { ...layer, shadows: 0 } });
    const unshaded = await capture("no-cloud-shadows");
    const shadowPixels = await differentPixels(page, [shaded, unshaded]);
    assert.ok(shadowPixels.changed > 20000, `clouds shade the ground: ${shadowPixels.changed} pixels`);
    // From above the layer the view's clouds lie over the ground below,
    // which the dome cannot show; off, only the dome shows them.
    await command("camera", { position: [0, 1900, 1], target: [400, 0, 300], vertical_fov_radians: 1.0, near: 0.5, far: 8000 });
    const over = (await command("environment", { clouds: layer })).result.rendering;
    assert.ok(over.clouds.view_blocks > 0, `the view marches its clouds: ${over.clouds.view_blocks}`);
    const below = await capture("clouds-below");
    const domed = (await command("environment", { clouds: { ...layer, in_view: false } })).result.rendering;
    assert.equal(domed.clouds.view_blocks, 0, "off, the view marches none");
    const domeOnly = await capture("clouds-dome-only");
    const viewPixels = await differentPixels(page, [below, domeOnly]);
    assert.ok(viewPixels.changed > 20000, `clouds lie below the camera: ${viewPixels.changed} pixels`);
    const blockless = await command("environment", { clouds: { ...layer, view_block: 0 } }, false);
    assert.equal(blockless.error.path, "$.clouds.view_block");
    const none = (await command("environment", { clouds: { ...layer, coverage: 0 } })).result.rendering;
    assert.equal(none.clouds.rows, 0, "no coverage marches nothing");
    const odd = await command("environment", { clouds: { ...layer, resolution: 1000 } }, false);
    assert.equal(odd.error.path, "$.clouds.resolution");
    await command("environment", { clouds: { ...layer, enabled: false } });
    await command("camera", overview);
    checks.push({ check: "clouds", rows: clouded.clouds.rows, pixels: cloudPixels.changed, shadows: shadowPixels.changed, view_blocks: over.clouds.view_blocks, below: viewPixels.changed });
  }
  // The physically based sky is opt-in: its tables of the
  // air build once, the sky-view again for each step of the sun, which the
  // time of day moves; it replaces the authored sky, a dusk sky is redder
  // than noon's, and settings out of range are refused.
  {
    await command("camera", { position: [0, 60, 0], target: [300, 200, 200], vertical_fov_radians: 1.0, near: 0.5, far: 4000 });
    const authored = await capture("sky-authored");
    const first = (await command("environment", { sky: {}, time_of_day: { hours: 12 } })).result.rendering;
    assert.equal(first.sky.tables, 3, `the air's tables and the sky-view build: ${first.sky.tables}`);
    assert.equal((await command("render")).result.rendering.sky.tables, 0, "unchanged, nothing builds");
    const noon = await capture("sky-noon");
    const replaced = await differentPixels(page, [authored, noon]);
    assert.ok(replaced.changed > 20000, `the physical sky replaces the authored one: ${replaced.changed} pixels`);
    const step = (await command("environment", { time_of_day: { hours: 19.3 } })).result.rendering;
    assert.equal(step.sky.tables, 1, `the sun's step builds the sky-view alone: ${step.sky.tables}`);
    const dusk = await capture("sky-dusk");
    const [noonColor, duskColor] = [await skyColor(page, noon), await skyColor(page, dusk)];
    assert.ok(duskColor[0] / Math.max(duskColor[2], 1) > noonColor[0] / Math.max(noonColor[2], 1),
      `dusk is redder than noon: ${noonColor} ${duskColor}`);
    const view = (await command("inspect", { component: "Sky" })).result.value;
    assert.equal(view.enabled, true);
    assert.equal((await command("environment", { sky: { mie_g: 2 } }, false)).error.path, "$.sky.mie_g");
    assert.equal((await command("environment", { time_of_day: { hours: 25 } }, false)).error.path, "$.time_of_day.hours");
    const off = (await command("environment", { sky: { enabled: false }, time_of_day: { enabled: false } })).result.rendering;
    assert.equal(off.sky.tables, 0, "without the sky nothing builds");
    await command("camera", overview);
    checks.push({ check: "sky", noon: noonColor.map(Math.round), dusk: duskColor.map(Math.round), replaced: replaced.changed });
  }

  await command("terrain", { view: "shaded" });
  await settle();

  // Long shadows: a low sun behind the dome, with shadow maps
  // reaching 60 m, shadows the plateau through the horizon map alone. A new
  // sun rebuilds the map, an unchanged one keeps it.
  const current = (await command("inspect", { component: "Environment" })).result.value;
  const atmosphere = { ...current.atmosphere, sun_direction: [1, 0.22, 1] };
  const shadows = { ...current.shadows, distance: 60, near_distance: 20 };
  // The control renders a frame with the new sun.
  const rebuilt = (await command("environment", { atmosphere, shadows })).result.rendering;
  // The terrain's and its surroundings'.
  assert.equal(rebuilt.terrain.horizons, 2);
  assert.ok(rebuilt.terrain.horizon_builds >= 1, `a new sun rebuilds: ${JSON.stringify(rebuilt)}`);
  assert.equal((await command("render")).result.rendering.terrain.horizon_builds, 0, "an unchanged sun keeps the map");
  const above = { ...overview, position: [-60, 320, 220], target: [0, 30, 0] };
  await command("camera", above);
  await settle();
  const long = await capture("horizon-shadows");
  const flat = await command("environment", { terrain: { ...fast, sun_shadows: false } });
  assert.equal(flat.result.rendering.terrain.horizons, 0);
  await settle();
  const short = await capture("map-shadows-only");
  const lengthened = await differentPixels(page, [long, short]);
  assert.ok(lengthened.changed > 20_000, `the dome's long shadow covers ${lengthened.changed} pixels`);
  checks.push({ check: "horizon shadows", builds: rebuilt.terrain.horizon_builds, pixels: lengthened.changed });

  // Aerial perspective: thick air from the eye hazes the
  // terrain; none leaves it as it was.
  await command("environment", { terrain: fast, atmosphere: { ...atmosphere, aerial_start: 0, aerial_density: 0 } });
  await settle();
  const clear = await capture("no-air");
  await command("environment", { atmosphere: { ...atmosphere, aerial_start: 0, aerial_density: 0.004 } });
  await settle();
  const hazed = await capture("thick-air");
  const air = await differentPixels(page, [clear, hazed]);
  assert.ok(air.changed > 100_000, `air hazes ${air.changed} pixels`);
  await command("environment", { atmosphere: current.atmosphere, shadows: current.shadows });
  await command("camera", overview);
  await settle();
  checks.push({ check: "aerial perspective", pixels: air.changed });

  // Running frames capture the probe's faces with terrain in them, once
  // the pipelines they draw with have compiled: polls while the page
  // compiles do not count against the capture's, up to three minutes.
  await command("resume");
  let probes;
  const resumed = Date.now();
  for (let attempt = 0; attempt < 50 && !(probes?.reflection_probes?.ready > 0) && Date.now() - resumed < 180_000;) {
    await new Promise(resolve => setTimeout(resolve, 200));
    probes = (await command("status")).result.rendering;
    if (!probes.pipelines_compiling) attempt++;
  }
  await command("pause");
  assert.ok(probes.reflection_probes.ready > 0, `probe captured: ${JSON.stringify(probes)}`);
  checks.push({ check: "probe captures", probes: probes.reflection_probes.ready });

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, checks }));
} finally {
  await close();
}
