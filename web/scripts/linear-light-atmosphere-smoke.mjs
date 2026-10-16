/** Radiance composition keeps water, ocean, clouds and physical sky drawing.
 * Uses the shared terrain fixture, checks feature state and fresh range
 * observations, and reports image changes without image threshold verdicts. */
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { differentPixels } from "./harness.mjs";
import { openTerrain, overview, skyColor } from "./terrain-fixture.mjs";

const { page, errors, command, close, capture, rendersUntil, output } = await openTerrain("linear-light-atmosphere");
const checks = [], measurements = [];

async function image(name) {
  const before = (await command("status")).result.rendering.radiance.source_frame ?? 0;
  const png = await capture(name);
  const fresh = await rendersUntil(stats => (stats.radiance.source_frame ?? 0) > before,
    `${name} must complete a fresh radiance observation`);
  assert.equal(fresh.radiance.limit_mask, 0, `${name} stays within the finite range`);
  checks.push({ name, radiance: fresh.radiance });
  return png;
}

async function difference(name, first, second) {
  const pixels = await differentPixels(page, [first, second]);
  measurements.push({ name, value: pixels.changed / pixels.total, unit: "fraction",
    note: "Changed pixels from the shared terrain fixture; an image observation, not a check verdict." });
}

try {
  await command("terrain", { view: "materials" });
  const water = (await command("inspect", { component: "Water" })).result.value;
  const wet = (await command("environment", { water })).result.rendering;
  assert.ok(wet.water.chunks > 0, "standing water draws chunks");
  const withWater = await image("water");
  await command("environment", { water: { ...water, enabled: false } });
  await difference("standing water changes", withWater, await image("water-disabled"));
  await command("environment", { water });

  const ocean = (await command("inspect", { component: "Ocean" })).result.value;
  await command("environment", { ocean: { ...ocean, enabled: true, level: 40 } });
  const sea = await image("ocean");
  const seaStats = (await command("render")).result.rendering.ocean;
  assert.ok(seaStats.sectors > 0 && seaStats.triangles > 0, "the ocean draws sectors");
  await command("environment", { ocean });
  await difference("ocean changes", sea, await image("ocean-disabled"));
  checks.push({ name: "water and ocean draw", chunks: wet.water.chunks, ocean: seaStats });

  const layer = { enabled: true, coverage: 0.6, base: 600, thickness: 900, density: 1,
    wind: [8, 3], scale: 2000, seed: 3, shadows: 0.75, resolution: 1024, slices: 16 };
  await command("camera", { position: [0, 60, 0], target: [300, 400, 200],
    vertical_fov_radians: 1, near: 0.5, far: 4000 });
  const clear = await image("sky-clear");
  const clouded = (await command("environment", { clouds: layer })).result.rendering;
  assert.equal(clouded.clouds.rows, 256, "exact frames march the whole cloud dome");
  const inspected = (await command("inspect", { component: "Clouds" })).result.value;
  assert.ok(Math.abs(inspected.coverage - layer.coverage) < 1e-6);
  await difference("cloud dome changes", clear, await image("sky-clouds"));

  await command("camera", { position: [0, 400, 1], target: [0, 30, 0],
    vertical_fov_radians: 1, near: 0.5, far: 4000 });
  const shadowed = await image("cloud-shadows");
  await command("environment", { clouds: { ...layer, shadows: 0 } });
  await difference("cloud shadow changes", shadowed, await image("cloud-shadows-disabled"));

  await command("camera", { position: [0, 1900, 1], target: [400, 0, 300],
    vertical_fov_radians: 1, near: 0.5, far: 8000 });
  const over = (await command("environment", { clouds: layer })).result.rendering;
  assert.ok(over.clouds.view_blocks > 0, "view clouds march above the ground");
  const below = await image("clouds-below");
  const domed = (await command("environment", { clouds: { ...layer, in_view: false } })).result.rendering;
  assert.equal(domed.clouds.view_blocks, 0, "disabled view clouds do not march");
  await difference("view cloud composition changes", below, await image("clouds-dome-only"));
  const empty = (await command("environment", { clouds: { ...layer, coverage: 0 } })).result.rendering;
  assert.equal(empty.clouds.rows, 0, "zero coverage does not march the dome");
  checks.push({ name: "clouds draw", rows: clouded.clouds.rows, view_blocks: over.clouds.view_blocks });
  await command("environment", { clouds: { ...layer, enabled: false } });

  await command("camera", { position: [0, 60, 0], target: [300, 200, 200],
    vertical_fov_radians: 1, near: 0.5, far: 4000 });
  const authored = await image("sky-authored");
  const first = (await command("environment", { sky: {}, time_of_day: { hours: 12 } })).result.rendering;
  assert.equal(first.sky.tables, 3, "the physical sky builds its three tables");
  assert.equal((await command("render")).result.rendering.sky.tables, 0, "unchanged sky keeps its tables");
  const noon = await image("sky-noon");
  await difference("physical sky changes", authored, noon);
  const step = (await command("environment", { time_of_day: { hours: 19.3 } })).result.rendering;
  assert.equal(step.sky.tables, 1, "a sun step rebuilds only the sky view");
  const dusk = await image("sky-dusk");
  await difference("time of day changes", noon, dusk);
  checks.push({ name: "physical sky draws", noon: await skyColor(page, noon), dusk: await skyColor(page, dusk) });
  await command("environment", { sky: { enabled: false }, time_of_day: { enabled: false } });
  await command("camera", overview);
  assert.deepEqual(errors, []);
  await writeFile(`${output}/results.json`, JSON.stringify({ ok: true, checks, measurements }, null, 2));
  console.log(JSON.stringify({ ok: true, output, checks, measurements }));
} catch (error) {
  await writeFile(`${output}/results.json`, JSON.stringify({ ok: false, error: String(error), checks, measurements, errors }, null, 2));
  if (error instanceof assert.AssertionError) { console.error(error); process.exitCode = 3; }
  else throw error;
} finally {
  await close();
}
