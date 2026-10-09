/**
 * Worker jobs: the Stress Lab's highlands load with their
 * bakes and tiles made on the page's job workers and their hiker's walks
 * searched there, every job's output the engine's own. Heights, normals,
 * holes, the tile levels answering and what the bakes and roads give at
 * points across the view match a session without workers (?workers=off)
 * bit for bit, as do the tiles, patches, water and objects placed on them
 * an exact frame of the same view draws; without workers the page hands
 * out no jobs. Screenshots of both views are kept to compare by eye: the
 * sky, wind and water move with time, which the sessions spend differently.
 */
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, devUrl, developmentReady, launch } from "./harness.mjs";

const output = await artifacts("jobs-smoke");
const { browser, page, errors } = await launch({ width: 960, height: 540, timeout: 600_000 });
// What an exact frame draws of the terrain, which time leaves alone.
const drawn = ["terrain_tiles", "terrain_patches", "terrain_triangles", "terrain_shadow_patches", "placed_objects", "placed_far_objects", "water_chunks"];
const command = commander(page, { deadline: 600_000 });
const view = { position: [-120, 0, 260], target: [380, 0, -340], vertical_fov_radians: 1.0, near: 0.2, far: 40000 };
// Points across the view, world x and z.
const points = [];
for (let row = 0; row < 16; row++) {
  for (let column = 0; column < 16; column++) points.push([-160 + column * 40, 300 - row * 45]);
}

/** Inspects resource `component`. */
async function inspect(component) {
  return (await command("inspect", { component })).result.value;
}

/** Renders until every wanted tile has been resident for a few frames and
 * nothing bakes; job outputs arrive between commands. */
async function settle(label) {
  let terrains;
  let quiet = 0;
  for (let attempt = 0; attempt < 600 && quiet < 3; attempt++) {
    await command("render");
    terrains = await inspect("Terrains");
    const settled = terrains.resident_tiles > 0 && terrains.baking === 0 && terrains.loading_tiles === 0 && terrains.fallback_patches === 0;
    quiet = settled ? quiet + 1 : 0;
    if (!settled) await delay(50);
  }
  assert.ok(quiet >= 3, `${label}: the terrain settles: ${JSON.stringify(terrains)}`);
  return terrains;
}

/** One session of the highlands with `query`: what it reports and samples;
 * with `walk`, once the hiker's first walk was found. */
async function session(name, query, walk) {
  const opened = Date.now();
  await page.goto(String(devUrl("stress.html", { preset: "highlands", prepare: "1", ...query })), { waitUntil: "domcontentloaded" });
  await developmentReady(page, { timeout: 600_000 });
  await page.waitForFunction(() => document.querySelector("canvas")?.style.visibility !== "hidden", null, { timeout: 600_000, polling: 100 });
  const shown = (Date.now() - opened) / 1000;
  // The hiker's first walk is searched while the view loads: at once on a
  // worker, over many ticks of their budget on the page.
  let agents;
  for (let attempt = 0; walk && attempt < 400; attempt++) {
    agents = await inspect("NavAgents");
    if (agents.walking > 0) break;
    await delay(50);
  }
  if (walk) {
    assert.equal(agents.walking, 1, `${name}: the hiker walks: ${JSON.stringify(agents)}`);
    assert.equal(agents.work, 0, `${name}: no tick searched on the page: ${JSON.stringify(agents)}`);
  }
  await command("pause");
  const [ground] = (await command("terrain_height", { points: [[view.position[0], view.position[2]]] })).result.samples;
  const camera = { ...view, position: [view.position[0], ground.height + 30, view.position[2]], target: [view.target[0], ground.height - 60, view.target[2]] };
  await command("camera", camera);
  const terrains = await settle(name);
  // Replies hold at most 64 KiB: sixty-four points at a time.
  const samples = [];
  for (let first = 0; first < points.length; first += 64) {
    samples.push(...(await command("terrain_height", { points: points.slice(first, first + 64) })).result.samples);
  }
  const rendering = (await command("render")).result.rendering;
  await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  const jobs = await inspect("Jobs");
  return { shown, terrains, samples, drawn: Object.fromEntries(drawn.map(field => [field, rendering[field]])), jobs, agents };
}

const checks = [];
try {
  const workers = await session("workers", {}, true);
  // The page's job cache writes its index a little after
  // its last change.
  await delay(3000);
  assert.ok(workers.jobs.workers > 0, `workers run jobs: ${JSON.stringify(workers.jobs)}`);
  assert.ok(workers.jobs.completed > 0 && workers.jobs.failed === 0, `the bakes and tiles ran there: ${JSON.stringify(workers.jobs)}`);
  assert.ok(workers.jobs.shares > 0 && workers.jobs.shared_bytes > 0, `the terrain's layers are shared with them: ${JSON.stringify(workers.jobs)}`);
  // Tiles hand out their shape before the rest.
  assert.ok(workers.jobs.parts > 0, `tiles hand out their shapes first: ${JSON.stringify(workers.jobs)}`);
  const page_ = await session("page", { workers: "off" }, false);
  assert.equal(page_.jobs.workers, 0, `?workers=off runs none: ${JSON.stringify(page_.jobs)}`);
  assert.equal(page_.jobs.submitted, 0, "without workers no job is handed out");
  assert.equal(workers.samples.length, page_.samples.length);
  const answered = workers.samples.filter(sample => sample !== null).length;
  assert.ok(answered > points.length / 2, `most points have ground: ${answered}`);
  assert.deepEqual(workers.samples, page_.samples, "heights, normals, holes, levels, bakes and roads match bit for bit");
  assert.ok(workers.drawn.placed_objects > 0, `objects stand on the tiles: ${JSON.stringify(workers.drawn)}`);
  assert.deepEqual(workers.drawn, page_.drawn, "an exact frame draws the same tiles, water and placed objects");
  // The first session kept its jobs' outputs; a session
  // verifying them takes them from the cache, runs every job anyway and
  // finds the same bytes, and samples and draws as the first did.
  const kept = await session("kept", { jobcache: "verify" }, true);
  assert.ok(kept.jobs.cache_hits > 0, `outputs come from the cache: ${JSON.stringify(kept.jobs)}`);
  assert.equal(kept.jobs.cache_mismatches, 0, `kept outputs match fresh runs: ${JSON.stringify(kept.jobs)}`);
  assert.deepEqual(kept.samples, workers.samples, "kept outputs sample as fresh ones");
  assert.deepEqual(kept.drawn, workers.drawn, "kept outputs draw as fresh ones");
  checks.push({ workers: { shown: workers.shown, jobs: workers.jobs, terrains: workers.terrains, drawn: workers.drawn },
    page: { shown: page_.shown, terrains: page_.terrains, drawn: page_.drawn },
    kept: { shown: kept.shown, jobs: kept.jobs }, answered });
  assert.deepEqual(errors, [], `page errors: ${errors.join("\n")}`);
  await writeFile(`${output}/report.json`, JSON.stringify(checks, null, 2));
  console.log(JSON.stringify({ ok: true, output, checks }, null, 2));
} finally {
  await browser.close();
}
