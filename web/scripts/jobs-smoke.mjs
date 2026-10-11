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
const drawn = ({ terrain, water }) => ({
  tiles: terrain.tiles, patches: terrain.patches, triangles: terrain.triangles, shadow_patches: terrain.shadow_patches,
  placed_objects: terrain.placed_objects, placed_far_objects: terrain.placed_far_objects, water_chunks: water.chunks,
});
const command = commander(page, { deadline: 600_000 });
// Observe the shipped runtime through its loaded binding; no Wasm or host
// implementation is replaced. A held real child keeps preparation pending.
await page.addInitScript(() => {
  if (location.hash !== "#bake-preparation") return;
  const trace = window.__bakePreparation = { calls: [], counts: {}, held: undefined, milestones: {}, busy: false };
  const post = Worker.prototype.postMessage;
  Worker.prototype.postMessage = function (message, transfer) {
    if (!trace.held && message.type === "run" && message.kind === "terrain_bake_child") {
      trace.held = { worker: this, message, transfer };
      trace.milestones.child = performance.now();
      return;
    }
    return post.call(this, message, transfer);
  };
  trace.release = () => {
    const { worker, message, transfer } = trace.held;
    post.call(worker, message, transfer);
  };
});
await page.route("**/stress_lab.js*", async route => {
  const response = await route.fetch();
  const instrumentation = `
if (window.__bakePreparation) {
  const trace = window.__bakePreparation;
  trace.milestones.module = performance.now();
  for (const name of ["set_effects", "set_job_workers", "work", "work_between_frames", "frame", "suspend", "take_jobs", "complete_job"]) {
    const original = StressGame.prototype[name];
    StressGame.prototype[name] = function (...args) {
      const preparing = document.getElementById("scene").style.visibility === "hidden";
      const before = ["work", "work_between_frames", "frame"].includes(name) ? JSON.parse(this.stats()) : undefined;
      trace.counts[name] = (trace.counts[name] || 0) + 1;
      if (trace.calls.length < 10000) trace.calls.push({ name, at: performance.now(), preparing, hidden: document.hidden,
        busy: trace.busy, args: ["work", "work_between_frames", "set_job_workers"].includes(name) ? args : undefined,
        compiling: before?.pipelines_compiling });
      const result = original.apply(this, args);
      if (name === "take_jobs" && result.some(message => message.kind === "terrain_bake")) trace.milestones.bake = performance.now();
      if (name === "frame") {
        const after = JSON.parse(this.stats());
        if (after.terrain.tiles > 0 && !trace.milestones.firstTile) trace.milestones.firstTile = performance.now();
      }
      return result;
    };
  }
}
`;
  await route.fulfill({ response, body: await response.text() + instrumentation });
});
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

/** Renders until every wanted tile has been resident for a few frames,
 * no job runs and the frames draw the same: job outputs, placement's
 * included, arrive between commands, later while the job cache is cold. */
async function settle(label) {
  let terrains;
  let quiet = 0;
  let previous;
  for (let attempt = 0; attempt < 600 && quiet < 3; attempt++) {
    const frame = JSON.stringify(drawn((await command("render")).result.rendering));
    terrains = await inspect("Terrains");
    const { running, bake } = await inspect("Jobs");
    const settled = terrains.resident_tiles > 0 && terrains.baking === 0 && terrains.loading_tiles === 0 && terrains.fallback_patches === 0
      && running === 0 && (!bake || bake.active_roots === 0) && frame === previous;
    previous = frame;
    quiet = settled ? quiet + 1 : 0;
    if (!settled) await delay(50);
  }
  assert.ok(quiet >= 3, `${label}: the terrain settles: ${JSON.stringify(terrains)}`);
  // However its tiles arrived, each holds the objects its level places.
  assert.equal(terrains.unplaced_tiles, 0, `${label}: every resident tile holds its objects: ${JSON.stringify(terrains)}`);
  return terrains;
}

/** One session of the highlands with `query`: what it reports and samples;
 * with `walk`, once the hiker's first walk was found. */
async function session(name, query, walk) {
  const opened = Date.now();
  await page.goto(String(devUrl("stress.html", { preset: "highlands", prepare: "1", ...(process.env.ENGINE_BUILD ? { engine_build: process.env.ENGINE_BUILD } : {}), ...query })), { waitUntil: "domcontentloaded" });
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
  return { shown, terrains, samples, drawn: drawn(rendering), jobs, agents };
}

/** Actual prepare=1 host, with one delayed child and observed method calls. */
async function preparation(between) {
  const url = devUrl("stress.html", { preset: "highlands", prepare: "1", workers: "2", jobcache: "off", between,
    ...(process.env.ENGINE_BUILD ? { engine_build: process.env.ENGINE_BUILD } : {}) });
  url.hash = "bake-preparation";
  await page.goto(String(url), { waitUntil: "domcontentloaded" });
  await developmentReady(page, { timeout: 600_000 });
  await page.waitForFunction(() => !!window.__bakePreparation?.held, null, { timeout: 600_000 });
  if (between === "on") await page.waitForFunction(() => window.__bakePreparation.calls.some(call => call.name === "work_between_frames" && call.preparing), null, { timeout: 600_000 });
  const first = await command("status");
  await delay(250);
  const second = await command("status");
  assert.equal(second.observation.tick, first.observation.tick, "preparation keeps simulation time suspended");
  await command("pause");
  const count = () => page.evaluate(() => [window.__bakePreparation.counts.work || 0, window.__bakePreparation.counts.work_between_frames || 0]);
  const paused = await count();
  await delay(250);
  assert.deepEqual(await count(), paused, "paused preparation owns no loading task");
  await command("resume");
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", { configurable: true, value: true });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  const hidden = await count();
  await delay(250);
  assert.deepEqual(await count(), hidden, "suspension stops extra preparation pumping");
  await page.evaluate(() => { delete document.hidden; document.dispatchEvent(new Event("visibilitychange")); });
  await command("pause");
  // Capture owns an asynchronous development command while the host is paused.
  await page.evaluate(async () => {
    const trace = window.__bakePreparation;
    trace.busy = true;
    try { await window.__engineDev.request({ op: "capture" }); } finally { trace.busy = false; }
  });
  await page.evaluate(() => window.__bakePreparation.release());
  await command("resume");
  await page.waitForFunction(() => document.getElementById("scene").style.visibility !== "hidden", null, { timeout: 600_000 });
  await command("pause");
  const trace = await page.evaluate(() => ({ calls: window.__bakePreparation.calls, counts: window.__bakePreparation.counts,
    milestones: { ...window.__bakePreparation.milestones, visible: performance.now() } }));
  const firstWork = trace.calls.findIndex(call => call.name === "work");
  assert.ok(firstWork > trace.calls.findIndex(call => call.name === "set_effects") && firstWork > trace.calls.findIndex(call => call.name === "set_job_workers"), "dispatch follows ready settings and pool attachment");
  assert.ok(trace.calls[firstWork].args.every(value => value > 1), "dispatch uses the applied viewport");
  const extra = trace.calls.filter(call => call.name === "work_between_frames" && call.preparing);
  if (between === "off") assert.equal(extra.length, 0, "between=off performs no extra preparation task");
  else assert.ok(extra.length > 0 && extra.every(call => call.args[0] === 2 && !call.compiling && !call.hidden && !call.busy), "extra tasks have a two-ms budget and exclude compilation, suspension and command ownership");
  assert.ok(trace.milestones.module < trace.milestones.bake && trace.milestones.bake <= trace.milestones.child
    && trace.milestones.child < trace.milestones.firstTile && trace.milestones.firstTile <= trace.milestones.visible, "recorded module, bake, child, first-tile and visible milestones are ordered");
  return { between, calls: trace.counts, extra: extra.length, milestones: trace.milestones };
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
  assert.ok(workers.jobs.bake?.completed_children > 0 && workers.jobs.bake.peak_busy_workers >= 2,
    `cold terrain baking assigns independent workers: ${JSON.stringify(workers.jobs.bake)}`);
  for (const field of ["active_roots", "children_in_flight", "private_share_bytes", "coordinator_bytes", "queued_bytes"]) {
    assert.equal(workers.jobs.bake[field], 0, `settled bake gauge ${field} is zero`);
  }
  assert.ok(workers.jobs.bake.sent_bytes > 0 && workers.jobs.bake.received_bytes > 0 && workers.jobs.bake.wasm_heap_bytes > 0,
    "bake transport and retained coordinator memory are observed");
  const described = await command("describe", { name: "Jobs" });
  assert.ok(JSON.stringify(described).includes("bake"), "describe Jobs documents optional bake diagnostics");
  const warm = await session("warm", {}, true);
  assert.ok(warm.jobs.cache_hits > 0, "a same-build warm session hits its isolated cache");
  assert.equal(warm.jobs.bake, undefined, "a cached bake creates no coordinator and schedules no children");
  assert.deepEqual(warm.samples, workers.samples, "warm and cold terrain samples agree");
  const one = await session("one-worker", { workers: "1", jobcache: "off", between: "off" }, true);
  assert.equal(one.jobs.bake, undefined, "one worker retains the serial bake path");
  assert.deepEqual(one.samples, workers.samples, "one-worker serial baking returns the same terrain");
  const two = await session("two-workers", { workers: "2", jobcache: "off", between: "on" }, true);
  assert.ok(two.jobs.bake?.completed_children > 0 && two.jobs.bake.peak_busy_workers === 2, "two workers use the graph");
  assert.deepEqual(two.samples, workers.samples, "two-worker baking returns the same terrain");
  assert.deepEqual(two.drawn, one.drawn, "bounded preparation preserves the visible terrain work");
  const page_ = await session("page", { workers: "off" }, false);
  assert.equal(page_.jobs.workers, 0, `?workers=off runs none: ${JSON.stringify(page_.jobs)}`);
  assert.equal(page_.jobs.submitted, 0, "without workers no job is handed out");
  assert.equal(page_.jobs.bake, undefined, "workers off creates no coordinator");
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
  assert.ok(kept.jobs.bake?.completed_children > 0, "cache verification runs a fresh complete graph");
  assert.deepEqual(kept.samples, workers.samples, "kept outputs sample as fresh ones");
  assert.deepEqual(kept.drawn, workers.drawn, "kept outputs draw as fresh ones");
  const preparationRuns = [await preparation("off"), await preparation("on")];
  checks.push({ workers: { shown: workers.shown, jobs: workers.jobs, terrains: workers.terrains, drawn: workers.drawn },
    page: { shown: page_.shown, terrains: page_.terrains, drawn: page_.drawn },
    warm: { shown: warm.shown, jobs: warm.jobs }, one: { shown: one.shown, jobs: one.jobs },
    two: { shown: two.shown, jobs: two.jobs }, kept: { shown: kept.shown, jobs: kept.jobs }, preparationRuns, answered });
  assert.deepEqual(errors, [], `page errors: ${errors.join("\n")}`);
  await writeFile(`${output}/report.json`, JSON.stringify(checks, null, 2));
  console.log(JSON.stringify({ ok: true, output, checks }, null, 2));
} catch (error) {
  process.exitCode = error?.code === "ERR_ASSERTION" ? 3 : 1;
  console.error(error);
} finally {
  await browser.close();
}
