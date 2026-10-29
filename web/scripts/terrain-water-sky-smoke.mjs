// @ts-check
import assert from "node:assert/strict";
import { differentPixels } from "./harness.mjs";
import { fast, openTerrain, overview, skyColor } from "./terrain-fixture.mjs";

const { page, errors, command, close, settle, capture, inspectGround } = await openTerrain("terrain-water-sky");
const checks = [];

try {
  const water = (await inspectGround()).hydrology;
  assert.equal(water.baking, false, "the bake is done once tiles load");
  assert.equal(water.samples, 513 * 513, `the bake reads its level: ${JSON.stringify(water)}`);
  assert.ok(water.channel_samples > 0 && water.max_area > 20000, `water gathers into channels: ${JSON.stringify(water)}`);
  const [watered] = (await command("query", "terrain_height", { points: [[-300, -300]] })).result.samples;
  assert.ok(watered.water && watered.water.flow > 0 && Number.isFinite(watered.water.wetness), `queries report water: ${JSON.stringify(watered)}`);
  assert.equal((await command("inspect", { component: "Terrains" })).result.value.falls, water.falls, `falls spawn as found: ${JSON.stringify(water)}`);
  checks.push({ check: "hydrology", samples: water.samples, channels: water.channel_samples, lakes: water.lake_samples, max_area: water.max_area, falls: water.falls });
  const grid = [];
  for (let z = -480; z <= 480; z += 40) for (let x = -480; x <= 480; x += 40) grid.push([x, z]);
  const lakes = [];
  for (let i = 0; i < grid.length; i += 100) {
    const samples = (await command("query", "terrain_height", { points: grid.slice(i, i + 100) })).result.samples;
    lakes.push(...samples.filter(sample => sample?.standing_water));
  }
  assert.ok(lakes.length > 0 && lakes.every(sample => sample.standing_water.depth > 0
    && Math.abs(sample.standing_water.surface - sample.standing_water.depth - sample.height) < 1e-2),
    `water stands in lakes: ${lakes.length}`);
  const waterSettings = (await command("inspect", { component: "Water" })).result.value;
  await command("control", "camera", overview);
  const wet = (await command("control", "environment", { water: waterSettings })).result.rendering;
  assert.ok(wet.water.leaves > 0, `water draws: ${wet.water.leaves}`);
  const withWater = await capture("water");
  await command("control", "environment", { water: { ...waterSettings, enabled: false } });
  const withoutWater = await capture("no-water");
  const waterPixels = await differentPixels(page, [withWater, withoutWater]);
  assert.ok(waterPixels.changed > 1000, `water shades ${waterPixels.changed} pixels`);
  const murky = await command("control", "environment", { water: { ...waterSettings, clarity: 0 } }, false);
  assert.equal(murky.error.path, "$.args.water.clarity");
  await command("control", "environment", { water: waterSettings });
  checks.push({ check: "water", lake_samples: lakes.length, leaves: wet.water.leaves, pixels: waterPixels.changed });
  {
    const [edge] = (await command("query", "terrain_height", { points: [[-280, 300]] })).result.samples;
    await command("control", "camera", { position: [-280, edge.height + 60, 300], target: [-280, edge.height, 300.01], vertical_fov_radians: 0.6, near: 0.5, far: 2000 });
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
    await command("control", "camera", overview);
    checks.push({ check: "interior", inside: dark, outside: light });
  }
  {
    const up = { position: [0, 60, 0], target: [300, 400, 200], vertical_fov_radians: 1.0, near: 0.5, far: 4000 };
    await command("control", "camera", up);
    const clear = await capture("sky-clear");
    const layer = { enabled: true, coverage: 0.6, base: 600, thickness: 900, density: 1, wind: [8, 3], scale: 2000, seed: 3, shadows: 0.75, resolution: 1024, slices: 16 };
    const clouded = (await command("control", "environment", { clouds: layer })).result.rendering;
    assert.equal(clouded.clouds.rows, 256, `exact frames march the whole dome: ${clouded.clouds.rows}`);
    const view = (await command("inspect", { component: "Clouds" })).result.value;
    assert.ok(Math.abs(view.coverage - 0.6) < 1e-6, `coverage reads back: ${view.coverage}`);
    const cloudy = await capture("sky-clouds");
    const cloudPixels = await differentPixels(page, [clear, cloudy]);
    assert.ok(cloudPixels.changed > 20000, `clouds show in the sky: ${cloudPixels.changed} pixels`);
    await command("control", "camera", { position: [0, 400, 1], target: [0, 30, 0], vertical_fov_radians: 1.0, near: 0.5, far: 4000 });
    const shaded = await capture("cloud-shadows");
    await command("control", "environment", { clouds: { ...layer, shadows: 0 } });
    const unshaded = await capture("no-cloud-shadows");
    const shadowPixels = await differentPixels(page, [shaded, unshaded]);
    assert.ok(shadowPixels.changed > 20000, `clouds shade the ground: ${shadowPixels.changed} pixels`);
    await command("control", "camera", { position: [0, 1900, 1], target: [400, 0, 300], vertical_fov_radians: 1.0, near: 0.5, far: 8000 });
    const over = (await command("control", "environment", { clouds: layer })).result.rendering;
    assert.ok(over.clouds.view_pixels > 0, `the view marches its clouds: ${over.clouds.view_pixels}`);
    const below = await capture("clouds-below");
    const domed = (await command("control", "environment", { clouds: { ...layer, in_view: false } })).result.rendering;
    assert.equal(domed.clouds.view_pixels, 0, "off, the view marches none");
    const domeOnly = await capture("clouds-dome-only");
    const viewPixels = await differentPixels(page, [below, domeOnly]);
    assert.ok(viewPixels.changed > 20000, `clouds lie below the camera: ${viewPixels.changed} pixels`);
    const none = (await command("control", "environment", { clouds: { ...layer, coverage: 0 } })).result.rendering;
    assert.equal(none.clouds.rows, 0, "no coverage marches nothing");
    const odd = await command("control", "environment", { clouds: { ...layer, resolution: 1000 } }, false);
    assert.equal(odd.error.path, "$.args.clouds.resolution");
    await command("control", "environment", { clouds: { ...layer, enabled: false } });
    await command("control", "camera", overview);
    checks.push({ check: "clouds", rows: clouded.clouds.rows, pixels: cloudPixels.changed, shadows: shadowPixels.changed, view_pixels: over.clouds.view_pixels, below: viewPixels.changed });
  }
  {
    await command("control", "camera", { position: [0, 60, 0], target: [300, 200, 200], vertical_fov_radians: 1.0, near: 0.5, far: 4000 });
    const authored = await capture("sky-authored");
    const first = (await command("control", "environment", { sky: {}, time_of_day: { hours: 12 } })).result.rendering;
    assert.equal(first.sky.tables, 3, `the air's tables and the sky-view build: ${first.sky.tables}`);
    assert.equal((await command("control", "render")).result.rendering.sky.tables, 0, "unchanged, nothing builds");
    const noon = await capture("sky-noon");
    const replaced = await differentPixels(page, [authored, noon]);
    assert.ok(replaced.changed > 20000, `the physical sky replaces the authored one: ${replaced.changed} pixels`);
    const step = (await command("control", "environment", { time_of_day: { hours: 19.3 } })).result.rendering;
    assert.equal(step.sky.tables, 1, `the sun's step builds the sky-view alone: ${step.sky.tables}`);
    const dusk = await capture("sky-dusk");
    const [noonColor, duskColor] = [await skyColor(page, noon), await skyColor(page, dusk)];
    assert.ok(duskColor[0] / Math.max(duskColor[2], 1) > noonColor[0] / Math.max(noonColor[2], 1),
      `dusk is redder than noon: ${noonColor} ${duskColor}`);
    await command("control", "environment", { time_of_day: { hours: 23 } });
    const starry = await capture("sky-night-sky");
    assert.equal((await command("inspect", { component: "NightSky" })).result.value.stars, 1);
    assert.equal((await command("inspect", { component: "Haze" })).result.value.enabled, true);
    await command("control", "environment", { night_sky: { stars: 0, moon: 0, cirrus: 0 } });
    const night = await capture("sky-night");
    const nightPixels = await differentPixels(page, [night, starry]);
    assert.ok(nightPixels.changed > 1000, `the night sky shows: ${nightPixels.changed} pixels`);
    await command("control", "environment", { night_sky: {} });
    assert.equal((await command("control", "environment", { night_sky: { cirrus: 2 } }, false)).error.path, "$.args.night_sky.cirrus");
    assert.equal((await command("control", "environment", { haze: { visibility: -1 } }, false)).error.path, "$.args.haze.visibility");
    let exposure = (await command("control", "environment", { auto_exposure: {} })).result.rendering.sky.exposure;
    assert.equal(exposure.active, true, "frames meter their exposure");
    for (let frame = 0; frame < 60 && exposure.metered === 0; frame++) {
      exposure = (await command("control", "render")).result.rendering.sky.exposure;
    }
    assert.ok(exposure.stops > 0 && exposure.stops <= 400, `the night's exposure within its limits: ${JSON.stringify(exposure)}`);
    const exposed = await capture("sky-night-exposed");
    const metered = await differentPixels(page, [starry, exposed]);
    assert.ok(metered.changed > 1000, `the meter's exposure replaces the sky's: ${metered.changed} pixels`);
    assert.equal((await command("control", "environment", { auto_exposure: { strength: 2 } }, false)).error.path, "$.args.auto_exposure.strength");
    await command("control", "environment", { auto_exposure: { enabled: false } });
    await command("control", "environment", { night_sky: { stars: 0, moon: 0, cirrus: 0 }, haze: { enabled: false } });
    const view = (await command("inspect", { component: "Sky" })).result.value;
    assert.equal(view.enabled, true);
    assert.equal((await command("control", "environment", { sky: { mie_g: 2 } }, false)).error.path, "$.args.sky.mie_g");
    assert.equal((await command("control", "environment", { time_of_day: { hours: 25 } }, false)).error.path, "$.args.time_of_day.hours");
    const off = (await command("control", "environment", { sky: { enabled: false }, time_of_day: { enabled: false } })).result.rendering;
    assert.equal(off.sky.tables, 0, "without the sky nothing builds");
    await command("control", "camera", overview);
    checks.push({ check: "sky", noon: noonColor.map(Math.round), dusk: duskColor.map(Math.round), replaced: replaced.changed, night_sky: nightPixels.changed, exposure: exposure.stops, exposed: metered.changed });
  }

  await settle();

  const current = (await command("inspect", { component: "Environment" })).result.value;
  const atmosphere = { ...current.atmosphere, sun_direction: [1, 0.22, 1] };
  const shadows = { ...current.shadows, levels: 1, finest_texel: 0.005 };
  const moved = (await command("control", "environment", { atmosphere, shadows })).result.rendering;
  assert.equal(moved.terrain.horizons, 2);
  assert.equal(moved.terrain.lighting_uploads, 0, `a new sun uploads nothing: ${JSON.stringify(moved)}`);
  const above = { ...overview, position: [-60, 320, 220], target: [0, 30, 0] };
  await command("control", "camera", above);
  await settle();
  const long = await capture("horizon-shadows");
  const flat = await command("control", "environment", { terrain: { ...fast, sun_shadows: false } });
  assert.equal(flat.result.rendering.terrain.horizons, 0);
  await settle();
  const short = await capture("map-shadows-only");
  const lengthened = await differentPixels(page, [long, short]);
  assert.ok(lengthened.changed > 20_000, `the dome's long shadow covers ${lengthened.changed} pixels`);
  checks.push({ check: "horizon shadows", horizons: moved.terrain.horizons, pixels: lengthened.changed });

  let bounced = (await command("control", "environment", { terrain: fast })).result.rendering;
  for (let frame = 0; frame < 400 && bounced.terrain.bounce.relit === 0; frame++) {
    await new Promise(resolve => setTimeout(resolve, 25));
    bounced = (await command("control", "render")).result.rendering;
  }
  assert.equal(bounced.terrain.bounce.relit, 1, `the terrain's probes relight: ${JSON.stringify(bounced.terrain)}`);
  assert.ok(bounced.terrain.bounce.probes > 0, "probes relight around the camera");
  await settle();
  const relit = await capture("bounce-light");
  const unlit = await command("control", "environment", { terrain: { ...fast, bounce: false } });
  assert.equal(unlit.result.rendering.terrain.bounce.relit, 0, "without bounce light nothing relights");
  await settle();
  const estimated = await capture("bounce-estimated");
  const bounce = await differentPixels(page, [relit, estimated]);
  assert.ok(bounce.changed > 1000, `bounce light shows: ${bounce.changed} pixels`);
  checks.push({ check: "bounce light", probes: bounced.terrain.bounce.probes, pixels: bounce.changed });

  await command("control", "environment", { terrain: fast, atmosphere: { ...atmosphere, aerial_start: 0, aerial_density: 0 } });
  await settle();
  const clear = await capture("no-air");
  await command("control", "environment", { atmosphere: { ...atmosphere, aerial_start: 0, aerial_density: 0.004 } });
  await settle();
  const hazed = await capture("thick-air");
  const air = await differentPixels(page, [clear, hazed]);
  assert.ok(air.changed > 100_000, `air hazes ${air.changed} pixels`);
  await command("control", "environment", { atmosphere: current.atmosphere, shadows: current.shadows });
  await command("control", "camera", overview);
  await settle();
  checks.push({ check: "aerial perspective", pixels: air.changed });

  await command("control", "resume");
  let probes;
  const resumed = Date.now();
  for (let attempt = 0; attempt < 50 && !(probes?.reflection_probes?.ready > 0) && Date.now() - resumed < 180_000;) {
    await new Promise(resolve => setTimeout(resolve, 200));
    probes = (await command("query", "status")).result.rendering;
    if (!probes.pipelines_compiling) attempt++;
  }
  await command("control", "pause");
  assert.ok(probes.reflection_probes.ready > 0, `probe captured: ${JSON.stringify(probes)}`);
  checks.push({ check: "probe captures", probes: probes.reflection_probes.ready });

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, checks }));
} finally {
  await close();
}
