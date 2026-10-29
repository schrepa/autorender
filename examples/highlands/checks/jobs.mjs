// @ts-check
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, devUrl, developmentReady, launch } from "../../../web/scripts/harness.mjs";

const output = await artifacts("highlands-jobs");
const { browser, page, errors } = await launch({ width: 960, height: 540, timeout: 600_000 });
const drawn = ({ terrain, water }) => ({
  tiles: terrain.tiles, leaves: terrain.leaves, triangles: terrain.triangles, shadow_patches: terrain.shadow_patches,
  placed_objects: terrain.placed_objects, placed_far_objects: terrain.placed_far_objects, water_patches: water.patches,
});
const command = commander(page, { deadline: 600_000 });
const view = { position: [-120, 0, 260], target: [380, 0, -340], vertical_fov_radians: 1.0, near: 0.2, far: 40000 };
const points = [];
for (let row = 0; row < 16; row++) {
  for (let column = 0; column < 16; column++) points.push([-160 + column * 40, 300 - row * 45]);
}

async function inspect(component) {
  return (await command("inspect", { component })).result.value;
}

async function settle(label) {
  let terrains;
  let quiet = 0;
  let previous;
  for (let attempt = 0; attempt < 600 && quiet < 3; attempt++) {
    const frame = JSON.stringify(drawn((await command("control", "render")).result.rendering));
    terrains = await inspect("Terrains");
    const { running } = await inspect("Jobs");
    const settled = terrains.resident_tiles > 0 && terrains.baking === 0 && terrains.loading_tiles === 0 && terrains.fallback_tiles === 0
      && running === 0 && frame === previous;
    previous = frame;
    quiet = settled ? quiet + 1 : 0;
    if (!settled) await delay(50);
  }
  assert.ok(quiet >= 3, `${label}: the terrain settles: ${JSON.stringify(terrains)}`);
  assert.equal(terrains.failed_jobs, 0, `${label}: no terrain job failed: ${terrains.latest_job_failure}`);
  return terrains;
}

async function session(name, query, walk) {
  const opened = Date.now();
  await page.goto(String(devUrl("highlands.html", { prepare: "1", ...query })), { waitUntil: "domcontentloaded" });
  await developmentReady(page, { timeout: 600_000 });
  await page.waitForFunction(() => document.querySelector("canvas")?.style.visibility !== "hidden", null, { timeout: 600_000, polling: 100 });
  const shown = (Date.now() - opened) / 1000;
  let agents;
  for (let attempt = 0; walk && attempt < 400; attempt++) {
    agents = await inspect("NavAgents");
    if (agents.walking > 0) break;
    await delay(50);
  }
  if (walk) {
    assert.equal(agents.walking, 1, `${name}: the hiker walks: ${JSON.stringify(agents)}`);
    assert.ok(agents.portals > 0 && agents.grounds > 0, `${name}: the walk read its tiles' portals and grounds: ${JSON.stringify(agents)}`);
    assert.equal(agents.failed_jobs, 0, `${name}: no record a walk read failed on a worker: ${agents.latest_job_failure}`);
  }
  await command("control", "pause");
  const [ground] = (await command("query", "terrain_height", { points: [[view.position[0], view.position[2]]] })).result.samples;
  const camera = { ...view, position: [view.position[0], ground.height + 30, view.position[2]], target: [view.target[0], ground.height - 60, view.target[2]] };
  await command("control", "camera", camera);
  const terrains = await settle(name);
  const samples = [];
  for (let first = 0; first < points.length; first += 64) {
    samples.push(...(await command("query", "terrain_height", { points: points.slice(first, first + 64) })).result.samples);
  }
  const rendering = (await command("control", "render")).result.rendering;
  await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  const jobs = await inspect("Jobs");
  return { shown, terrains, samples, drawn: drawn(rendering), jobs, agents };
}

async function cacheIndexWritten() {
  for (let attempt = 0; ; attempt++) {
    const { listed, held } = await page.evaluate(async () => {
      try {
        const folder = await (await navigator.storage.getDirectory()).getDirectoryHandle("engine-job-cache");
        const held = [];
        for await (const name of folder.keys()) if (name !== "index.json") held.push(name);
        const index = JSON.parse(await (await (await folder.getFileHandle("index.json")).getFile()).text());
        return { listed: Object.keys(index).sort(), held: held.sort() };
      } catch {
        return { listed: [], held: [] };
      }
    });
    if (held.length > 0 && JSON.stringify(listed) === JSON.stringify(held)) return;
    assert.ok(attempt < 600, `the job cache's index lists what its folder holds: ${listed.length} of ${held.length} outputs`);
    await delay(50);
  }
}

const checks = [];
try {
  const workers = await session("workers", {}, true);
  await cacheIndexWritten();
  assert.ok(workers.jobs.workers > 0, `workers run jobs: ${JSON.stringify(workers.jobs)}`);
  assert.ok(workers.jobs.completed > 0 && workers.jobs.failed === 0, `the bakes and tiles ran there: ${JSON.stringify(workers.jobs)}`);
  assert.ok(workers.jobs.shares > 0 && workers.jobs.shared_bytes > 0, `the terrain's layers are shared with them: ${JSON.stringify(workers.jobs)}`);
  assert.equal(workers.jobs.in_place, 0, `with workers no job runs in place: ${JSON.stringify(workers.jobs)}`);
  const page_ = await session("page", { workers: "off", jobcache: "off" }, false);
  assert.equal(page_.jobs.workers, 0, `?workers=off runs none: ${JSON.stringify(page_.jobs)}`);
  assert.ok(page_.jobs.in_place > 0 && page_.jobs.failed === 0, `without workers the page runs its jobs in place: ${JSON.stringify(page_.jobs)}`);
  assert.equal(workers.samples.length, page_.samples.length);
  const answered = workers.samples.filter(sample => sample !== null).length;
  assert.ok(answered > points.length / 2, `most points have ground: ${answered}`);
  assert.deepEqual(workers.samples, page_.samples, "heights, normals, holes, levels, bakes and roads match bit for bit");
  assert.ok(workers.drawn.placed_objects > 0, `objects stand on the tiles: ${JSON.stringify(workers.drawn)}`);
  assert.deepEqual(workers.drawn, page_.drawn, "an exact frame draws the same tiles, water and placed objects");
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
