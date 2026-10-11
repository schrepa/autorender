/** Full terrain bake bytes and lifecycle: the shipped coordinator and JobWorker,
 * including Highlands, tile edges, ordered layers, reversed child delivery,
 * malformed results, cancellation and retained coordinator heap capacity.
 */
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, devUrl, developmentReady, launch, openPaused, sceneUrl, routedScene } from "./harness.mjs";

const output = await artifacts("bake-scheduling-smoke");
const { browser, page, errors } = await launch({ width: 320, height: 240, timeout: 600_000 });
page.on("console", entry => { if (entry.text().startsWith("bake-scheduling:")) console.log(entry.text()); });
page.on("pageerror", error => console.error(error));
const bindingUrls = new Set();
page.on("response", response => {
  if (response.request().resourceType() === "script" && new URL(response.url()).pathname.endsWith("/stress_lab.js")) bindingUrls.add(response.url());
});
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
  // This is binding evaluation, not completion of Wasm compilation.
  trace.milestones.bindingEvaluation = performance.now();
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
  assert.ok(trace.milestones.bindingEvaluation < trace.milestones.bake && trace.milestones.bake <= trace.milestones.child
    && trace.milestones.child < trace.milestones.firstTile && trace.milestones.firstTile <= trace.milestones.visible, "recorded binding evaluation, bake, child, first-tile and visible milestones are ordered");
  return { between, calls: trace.counts, extra: extra.length, milestones: trace.milestones };
}


async function hostChecks() {
  const checks = [];
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
  return checks;
}

const empty = {
  version: 9, id: "bake-input-fixture", revision: 1,
  camera: { position: [0, 20, 40], target: [0, 0, 0], fov_degrees: 60, near: 0.1, far: 2000 },
  materials: [{ id: "plain", color: [0.4, 0.5, 0.6, 1] }],
  meshes: [{ id: "box", source: { kind: "cube" } }],
  entities: [{ id: "player", mesh: "box", material: "plain", tags: ["player"],
    collider: { shape: { kind: "cuboid", half_extents: [0.5, 0.5, 0.5] }, memberships: 2, filter: 1, sensor: false },
    transform: { translation: [0, 10, 0] } }],
};
const scene = routedScene(page, empty);
const highlands = JSON.parse(await readFile(new URL("../../bench/scenes/highlands.json", import.meta.url), "utf8")).options.config;
const fixtures = [
  ["flat", 0, 0, false], ["rough", 120, 0, false], ["basin", 30, 0, true], ["negative", 120, -180, false],
].map(([name, height, offset, basin]) => ({ name, scene: {
  ...empty, id: `bake-${name}`, entities: [...empty.entities, { id: "ground", terrain: {
    tiles: [2, 2], spacing: 1, origin: [-256, 0, -256],
    layers: [
      { noise: { seed: 1337, height, offset, wavelength: 180 } },
      ...(basin ? [{ stamp: { center: [256, 256], radius: 120, falloff: 60, op: { lower: 70 } } }] : []),
      { erosion: { level: 0, rounds: 2, incision: 0.002, creep: 2 } },
      { hydrology: { level: 0, channel_area: 1000, stream_area: 200, depth: 1, width: 4, max_depth: 3, max_width: 20 } },
      { path: { points: [[10, 10], [260, 260], [500, 490]], width: 6, shoulder: 6, smoothing: 60 } },
      { path: { points: [[10, 480], [260, 260], [480, 10]], width: 6, shoulder: 6, smoothing: 60 } },
    ],
    surface: { materials: [{ id: "ground", color: [0.2, 0.3, 0.1] }], biomes: [{ id: "all", layers: [{ material: "ground" }] }] },
  } }],
} }));

// The same chained fixture also runs through the real host and live reload.
// Interior points straddle the duplicated tile edge at world coordinate zero.
const liveFixture = structuredClone(fixtures.find(fixture => fixture.name === "rough").scene);
liveFixture.id = "bake-live-edges";
const liveLayers = liveFixture.entities.find(entity => entity.id === "ground").terrain.layers;
liveLayers.splice(3, 0, { stamp: { center: [260, 260], radius: 10, falloff: 10, op: { lower: 12 } } });
for (const layer of liveLayers) if (layer.path) { layer.path.bridge_fill = 4; layer.path.tunnel_cut = 25; }

async function liveEdges() {
  scene.serve(liveFixture);
  const query = { workers: "2", jobcache: "off", ...(process.env.ENGINE_BUILD ? { engine_build: process.env.ENGINE_BUILD } : {}) };
  await openPaused(page, sceneUrl(query), { timeout: 600_000 });
  const camera = { position: [0, 200, 300], target: [0, 0, 0], vertical_fov_radians: Math.PI / 3, near: 0.5, far: 2000 };
  const prepare = async label => {
    await command("environment", { terrain: { samples_per_frame: 50_000_000 } });
    await command("camera", camera);
    await settle(label);
  };
  const points = [-128, -1, 0, 1, 128].flatMap(x => [[x, 0], [0, x]]);
  const samples = async () => (await command("terrain_height", { points })).result.samples;
  const terrain = async () => (await command("inspect", { entity: "ground", component: "Terrain" })).result.entity.components.Terrain;
  await prepare("live edges");
  const before = await terrain(), edgeSamples = await samples();
  assert.ok(edgeSamples.every(sample => sample && Number.isFinite(sample.height)), "tile-edge samples are finite");
  assert.ok(before.hydrology && before.paths.length === 2, "the live fixture reaches its hydrology and both path bakes");
  const jobsBefore = await inspect("Jobs");
  assert.ok(jobsBefore.bake?.completed_children > 0, "the live fixture used the parallel bake graph");
  await scene.reload(command, structuredClone(liveFixture));
  await prepare("live revision");
  assert.deepEqual(await samples(), edgeSamples, "fresh parallel joins preserve duplicate-edge samples");
  const after = await terrain();
  assert.deepEqual(after.hydrology, before.hydrology, "water survives live bake revision replacement");
  assert.deepEqual(after.paths, before.paths, "road and bridge profiles survive live bake revision replacement");
  const jobsAfter = await inspect("Jobs");
  for (const field of ["active_roots", "children_in_flight", "private_share_bytes", "coordinator_bytes", "queued_bytes"]) {
    assert.equal(jobsAfter.bake[field], 0, `replaced scene releases bake gauge ${field}`);
  }
  return { points, edgeSamples, hydrology: before.hydrology, paths: before.paths, jobsBefore, jobsAfter, revision: scene.revision };
}

try {
  await scene.route();
  const query = { workers: "off", jobcache: "off", effects: "none", ...(process.env.ENGINE_BUILD ? { engine_build: process.env.ENGINE_BUILD } : {}) };
  await openPaused(page, sceneUrl(query), { timeout: 600_000 });
  assert.equal(bindingUrls.size, 1, "one exact loaded game build");
  const moduleUrl = [...bindingUrls][0];
  if (process.env.ENGINE_BUILD) assert.ok(moduleUrl.includes(`/${process.env.ENGINE_BUILD}/`), "the requested build is pinned");
  const graph = await page.evaluate(async ({ moduleUrl, fixtures, highlands }) => {
    const gameModule = await import(moduleUrl);
    const workerUrl = new URL("engine_worker.js", moduleUrl).href;
    const workerModule = await import(workerUrl);
    const instance = await workerModule.default();
    const check = (condition, message) => { if (!condition) throw new Error(`ASSERT: ${message}`); };
    const hash = async bytes => [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map(byte => byte.toString(16).padStart(2, "0")).join("");
    const whole = (worker, kind, input, id = 1) => {
      worker.start(id, kind, input);
      for (let steps = 0; steps < 1_000_000; steps++) {
        const output = worker.step(id, 65536);
        if (output !== undefined) return output;
      }
      throw new Error("ASSERT: worker did not finish within its step bound");
    };
    async function inputOf(fixture, config) {
      const canvas = document.createElement("canvas");
      canvas.width = 32; canvas.height = 32;
      document.body.append(canvas);
      let game;
      try {
        game = config ? await gameModule.StressGame.create(canvas, JSON.stringify(config), undefined)
          : await gameModule.StressGame.create_scene(canvas, JSON.stringify(fixture), new gameModule.SceneFileSet(), "");
        game.set_job_workers(2);
        for (let attempt = 0; attempt < 20; attempt++) {
          game.work(32, 32);
          for (const message of game.take_jobs()) if (message.kind === "terrain_bake") return message.input;
        }
        throw new Error("ASSERT: the real game must submit a terrain_bake before any inline bake");
      } finally { game?.free(); canvas.remove(); }
    }
    let epoch = 1;
    function graphBytes(input, count, order) {
      const graph = new workerModule.BakeCoordinator(input, epoch++);
      const workers = Array.from({ length: count }, () => new workerModule.JobWorker());
      const shares = new Set();
      let children = 0;
      try {
        for (let stage = 0; stage < 100_000; stage++) {
          const done = [];
          for (const action of graph.ready(count * 2)) {
            if (action.share !== undefined) {
              shares.add(action.share);
              workers.forEach(worker => worker.share(action.share, action.bytes));
            } else if (action.unshare !== undefined) {
              shares.delete(action.unshare);
              workers.forEach(worker => worker.unshare(action.unshare));
            } else {
              const output = whole(workers[children++ % count], "terrain_bake_child", action.input, action.id);
              done.push({ id: action.id, output });
            }
          }
          if (order === "reverse") done.reverse();
          else if (order === "shuffle") done.sort((left, right) => ((left.id * 2654435761) >>> 0) - ((right.id * 2654435761) >>> 0));
          done.forEach(({ id, output }) => graph.complete(id, output));
          const bytes = graph.take_result();
          if (bytes !== undefined) {
            for (const action of graph.ready(0)) {
              check(action.unshare !== undefined, "a finished root only releases shares");
              shares.delete(action.unshare);
              workers.forEach(worker => worker.unshare(action.unshare));
            }
            check(shares.size === 0, "a finished root releases every private share");
            const stats = Array.from(graph.diagnostics());
            check(stats[0] === 0 && stats[4] === 0 && stats[5] === 0, "a finished root retains no joins or shares");
            return { bytes, children, stats };
          }
        }
        throw new Error("ASSERT: graph did not finish within the stage bound");
      } finally {
        graph.cancel(); graph.free();
        workers.forEach(worker => worker.free());
      }
    }
    const inputs = [];
    for (const fixture of fixtures) inputs.push({ name: fixture.name, input: await inputOf(fixture.scene) });
    inputs.push({ name: "highlands", input: await inputOf(undefined, highlands) });
    const changed = structuredClone(fixtures[1].scene);
    changed.entities.find(entity => entity.id === "ground").terrain.layers[0].noise.seed++;
    inputs.push({ name: "changed-lower-layer", input: await inputOf(changed) });
    const results = [];
    for (const { name, input } of inputs) {
      const serialWorker = new workerModule.JobWorker();
      let serial;
      try { serial = whole(serialWorker, "terrain_bake", input); } finally { serialWorker.free(); }
      const expected = await hash(serial);
      const variants = [];
      for (const [workers, order] of [[1, "ordered"], [2, "reverse"], [Math.max(2, Math.min(8, Math.floor((navigator.hardwareConcurrency || 4) / 2))), "shuffle"]]) {
        const candidate = graphBytes(input, workers, order);
        check(candidate.bytes.length === serial.length, `${name}: complete output length at ${workers} workers`);
        check(candidate.bytes.every((byte, index) => byte === serial[index]), `${name}: complete serial byte equivalence at ${workers} workers/${order}`);
        variants.push({ workers, order, hash: await hash(candidate.bytes), children: candidate.children, diagnostics: candidate.stats });
      }
      results.push({ name, inputBytes: input.length, bytes: serial.length, hash: expected, variants });
      console.log(`bake-scheduling: ${name}, ${serial.length} bytes, all worker/order variants match`);
    }
    check(results.find(value => value.name === "rough").hash !== results.find(value => value.name === "changed-lower-layer").hash,
      "changing a lower layer changes the final expected bytes");

    const input = inputs[1].input;
    // Exercise the shipped host pool and real Web Workers with a bounded
    // logical-runtime fixture. Transport interception belongs to this
    // check, so the engine needs no fault-injection API.
    const { JobPool } = await import("/src/engine/jobs.ts");
    const until = async (condition, label) => {
      const deadline = performance.now() + 120_000;
      while (!condition()) {
        check(performance.now() < deadline, `${label}: bounded completion`);
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    };
    const runtime = () => {
      const queued = [];
      const completed = [], failed = [], snapshots = [];
      return {
        queued, completed, failed, snapshots, workers: 0,
        set_job_workers(value) { this.workers = value; },
        take_jobs() { return queued.splice(0); },
        complete_job(id, bytes) { completed.push({ id, bytes }); },
        fail_job(id, error) { failed.push({ id, error }); },
        part_job() { throw new Error("ASSERT: bake children never become logical partial outputs"); },
        report_job_cache() {},
        report_bake_jobs(json) { snapshots.push(JSON.parse(json)); },
      };
    };
    const pools = [];
    {
      // Hold two real host cache lookups across a runtime replacement with
      // the same logical ID. The old lookup must not erase the new one's
      // cancellation token when its promise settles later.
      const pool = new JobPool(2), old = runtime(), current = runtime(), lookups = [];
      pool.cache = Promise.resolve({
        key: async () => "bake-cancel-fixture", lookup: () => new Promise(resolve => lookups.push(resolve)),
        stats: () => ({ hits: 0, misses: 0, bytes: 0, mismatches: 0 }), verify: false,
      });
      try {
        pool.attach(old);
        old.queued.push({ id: 100, kind: "terrain_bake", input: input.slice(), priority: 0, identity: new Uint8Array([1]) });
        pool.exchange();
        await until(() => lookups.length === 1, "old runtime lookup");
        pool.attach(current);
        current.queued.push({ id: 100, kind: "terrain_bake", input: input.slice(), priority: 0, identity: new Uint8Array([1]) });
        pool.exchange();
        await until(() => lookups.length === 2, "replacement runtime lookup");
        lookups[0](undefined);
        await new Promise(resolve => setTimeout(resolve, 20));
        current.queued.push({ cancel: 100 }); pool.exchange();
        lookups[1](undefined);
        await new Promise(resolve => setTimeout(resolve, 100));
        check(old.completed.length + old.failed.length + current.completed.length + current.failed.length === 0,
          "old and cancelled cache lookups never dispatch or complete");
        check(pool.bakes.size === 0 && pool.starting.size === 0 && pool.routes.size === 0 && pool.shares.size === 0,
          "cache lookup cancellation leaves no graph, transport or private share");
        pools.push({ mode: "cache-cancel", completed: [], failed: [] });
      } finally { pool.detach(); for (const slot of pool.slots) slot.worker.terminate(); }
    }
    for (const mode of ["one", "two", "cancel", "replace", "worker-loss"]) {
      const pool = new JobPool(mode === "one" ? 1 : 2);
      const first = runtime();
      let held;
      const fault = ["cancel", "replace", "worker-loss"].includes(mode);
      for (const slot of pool.slots) {
        const post = slot.worker.postMessage.bind(slot.worker);
        slot.worker.postMessage = (message, transfer) => {
          if (fault && !held && message.type === "run" && message.kind === "terrain_bake_child") {
            held = { slot, post, message, transfer };
            return;
          }
          post(message, transfer);
        };
      }
      try {
        pool.attach(first);
        first.queued.push({ clear: true }, { id: 100, kind: "terrain_bake", input: input.slice(), priority: 0 });
        pool.exchange();
        if (fault) {
          await until(() => held !== undefined, `${mode}: child assigned`);
          if (mode === "cancel") {
            first.queued.push({ cancel: 100 }); pool.exchange();
            held.post(held.message, held.transfer);
            await new Promise(resolve => setTimeout(resolve, 250));
            check(first.completed.length === 0 && first.failed.length === 0, "cancelled logical request has no late completion/failure");
          } else if (mode === "replace") {
            const replacement = runtime();
            pool.attach(replacement);
            replacement.queued.push({ clear: true }, { id: 100, kind: "terrain_bake", input: input.slice(), priority: 0 });
            pool.exchange();
            held.post(held.message, held.transfer);
            await until(() => replacement.completed.length + replacement.failed.length > 0, "replacement runtime");
            check(replacement.failed.length === 0 && replacement.completed.length === 1, "replacement completes once");
            check(await hash(replacement.completed[0].bytes) === results.find(value => value.name === "rough").hash, "replacement gets current complete serial bytes");
            check(first.completed.length === 0 && first.failed.length === 0, "disposed runtime receives no old child result");
          } else {
            held.slot.worker.terminate();
            held.slot.worker.onerror({ preventDefault() {}, message: "intentional bake smoke worker loss" });
            await until(() => first.failed.length > 0, "lost worker fails root");
            check(first.failed.length === 1 && first.completed.length === 0, "worker loss fails the logical root exactly once");
            first.queued.push({ id: 101, kind: "terrain_bake", input: input.slice(), priority: 0 });
            pool.exchange();
            await until(() => first.completed.length > 0 || first.failed.length > 1, "replacement worker recovery");
            check(first.failed.length === 1 && first.completed.length === 1, "replacement pool completes a fresh root");
            check(await hash(first.completed[0].bytes) === results.find(value => value.name === "rough").hash, "recovery returns complete serial bytes");
          }
        } else {
          await until(() => first.completed.length + first.failed.length > 0, `${mode}: live pool`);
          check(first.failed.length === 0 && first.completed.length === 1, "live pool completes the root once");
          check(await hash(first.completed[0].bytes) === results.find(value => value.name === "rough").hash, "live pool matches serial bytes");
          const diagnostics = first.snapshots.filter(Boolean).at(-1);
          if (mode === "one") check(diagnostics === undefined, "one worker uses no coordinator");
          else check(diagnostics.completed_children > 0 && diagnostics.peak_busy_workers >= 2, "two workers are assigned independent children");
        }
        pool.exchange();
        const stats = first.snapshots.filter(Boolean).at(-1);
        if (stats && mode !== "replace") check(stats.active_roots === 0 && stats.children_in_flight === 0 && stats.private_share_bytes === 0 && stats.coordinator_bytes === 0, "idle graph gauges are zero");
        check(pool.bakes.size === 0 && pool.starting.size === 0 && pool.routes.size === 0 && pool.shares.size === 0,
          `${mode}: no live graph, transport or private share remains`);
        pools.push({ mode, completed: first.completed.map(value => value.id), failed: first.failed, stats });
      } finally {
        pool.detach();
        for (const slot of pool.slots) slot.worker.terminate();
      }
    }
    const rejects = [];
    const thrown = action => { try { action(); return false; } catch { return true; } };
    for (const mode of ["erosion-spacing", "erosion-dimensions", "hydrology-spacing", "river-grid", "path-point", "path-range", "river-range"]) {
      const graph = new workerModule.BakeCoordinator(input, epoch++), worker = new workerModule.JobWorker();
      let refused = false;
      try {
        for (let stage = 0; stage < 1000 && !refused; stage++) {
          for (const action of graph.ready(2)) {
            if (action.share !== undefined) worker.share(action.share, action.bytes);
            else if (action.unshare !== undefined) worker.unshare(action.unshare);
            else {
              const output = whole(worker, "terrain_bake_child", action.input, action.id);
              const final = output[4] === 4;
              const target = mode.startsWith("erosion") ? final && output[5] === 1
                : ["hydrology-spacing", "river-grid"].includes(mode) ? final && output[5] === 0
                : mode === "path-point" ? final && output[5] === 2
                : mode === "path-range" ? action.input[4] === 1 : action.input[4] === 2;
              if (!target) { graph.complete(action.id, output); continue; }
              const view = new DataView(output.buffer, output.byteOffset, output.byteLength);
              if (mode === "erosion-dimensions") view.setUint32(10, 1, true);
              else if (mode.endsWith("spacing")) view.setFloat32(6, NaN, true);
              else if (mode === "path-point") view.setFloat32(14, Infinity, true);
              else if (mode === "river-grid") {
                // Skip the version-1 hydrology images and water arrays to
                // its first river point. A finite extreme must be refused
                // before the decoder allocates the derived spatial grid.
                let cursor = 10;
                const count = () => { const value = Number(view.getBigUint64(cursor, true)); cursor += 8; return value; };
                const vector = bytes => { const length = count(); cursor += length * bytes; };
                const image = () => {
                  cursor += 8; vector(2);
                  const levels = count();
                  for (let level = 0; level < levels; level++) { cursor += 8; vector(4); }
                };
                image(); cursor += 4; image(); image(); image();
                cursor += 20; vector(4); vector(4); vector(8);
                check(count() > 0 && count() >= 2, "rough hydrology has a river segment");
                view.setFloat32(cursor, 3.4028234663852886e38, true);
              }
              else view.setBigUint64(5, view.getBigUint64(5, true) + 1n, true);
              check(thrown(() => graph.complete(action.id, output)), `${mode}: malformed terminal metadata or overlapping range is refused`);
              check(graph.take_result() === undefined, `${mode}: no malformed baked layer is exposed`);
              graph.cancel();
              for (const release of graph.ready(0)) if (release.unshare !== undefined) worker.unshare(release.unshare);
              refused = true;
              break;
            }
          }
        }
        check(refused, `${mode}: the fixture reaches the intended stage`);
        rejects.push(mode);
      } finally { graph.free(); worker.free(); }
    }
    {
      const graph = new workerModule.BakeCoordinator(input, epoch++), worker = new workerModule.JobWorker(), shares = new Set();
      let boundary = false;
      try {
        for (let stage = 0; stage < 1000 && !boundary; stage++) {
          for (const action of graph.ready(2)) {
            if (action.share !== undefined) { shares.add(action.share); worker.share(action.share, action.bytes); }
            else if (action.unshare !== undefined) { shares.delete(action.unshare); worker.unshare(action.unshare); }
            else {
              const output = whole(worker, "terrain_bake_child", action.input, action.id);
              graph.complete(action.id, output);
              if (output[4] === 4) { boundary = true; break; }
            }
          }
        }
        check(boundary, "fixture reaches the transition between ordered layers");
        graph.cancel();
        for (const action of graph.ready(0)) {
          check(action.unshare !== undefined, "cancellation drops a queued next-layer share");
          shares.delete(action.unshare); worker.unshare(action.unshare);
        }
        check(shares.size === 0, "cancellation retains the previous layer's queued unshare");
        rejects.push("layer-boundary-cancel");
      } finally { graph.free(); worker.free(); }
    }
    for (const epoch of [NaN, Infinity, -1, 0.5, Number.MAX_SAFE_INTEGER + 1]) check(thrown(() => new workerModule.BakeCoordinator(input, epoch)), "invalid numeric epoch rejected");
    for (const mode of ["short", "version", "tile-key", "nan", "duplicate", "unexpected", "cancel", "missing"]) {
      const graph = new workerModule.BakeCoordinator(input, epoch++);
      const worker = new workerModule.JobWorker();
      try {
        const actions = graph.ready(2);
        for (const action of actions) if (action.share !== undefined) worker.share(action.share, action.bytes);
        const child = actions.find(action => action.id !== undefined);
        check(!!child, "fixture has a real child");
        const output = whole(worker, "terrain_bake_child", child.input, child.id);
        let malformed = output.slice();
        if (mode === "short") malformed = malformed.slice(0, -1);
        if (mode === "version") malformed[0] ^= 0x40;
        if (mode === "tile-key") malformed[6] ^= 0x40;
        // Tile output: version u32, tag u8, TileKey (u8,u16,u16),
        // count u64, then the f32 interior samples.
        if (mode === "nan") new DataView(malformed.buffer).setFloat32(18, NaN, true);
        if (mode === "duplicate") graph.complete(child.id, output);
        if (mode === "cancel" || mode === "missing") {
          check(graph.take_result() === undefined, "a missing child cannot publish a partial layer");
          graph.cancel();
        }
        check(thrown(() => graph.complete(mode === "unexpected" ? child.id + 100000 : child.id, malformed)), `${mode}: invalid/late result is refused`);
        graph.cancel(); // idempotent; must retain the pending unshare action.
        check(graph.take_result() === undefined, `${mode}: no partial result`);
        check(graph.ready(0).some(action => action.unshare !== undefined), `${mode}: private share released exactly once`);
        check(graph.ready(0).length === 0, `${mode}: no repeated release`);
        check(Array.from(graph.diagnostics()).every((value, index) => ![0, 4, 5].includes(index) || value === 0), `${mode}: no live graph buffers`);
        rejects.push(mode);
      } finally { graph.free(); worker.free(); }
    }
    const heap = [];
    for (let cycle = 0; cycle < 12; cycle++) {
      const graph = new workerModule.BakeCoordinator(input, epoch++);
      graph.ready(2); graph.cancel(); graph.ready(0); graph.free();
      heap.push(instance.memory.buffer.byteLength);
    }
    check(new Set(heap.slice(-5)).size === 1, "retained coordinator Wasm capacity stabilizes after repeated cancellation");
    return { moduleUrl, workerUrl, results, pools, rejects, heap };
  }, { moduleUrl, fixtures: [...fixtures, { name: "live-edges", scene: liveFixture }], highlands });
  await writeFile(`${output}/graph.json`, JSON.stringify(graph, null, 2));
  console.log("bake-scheduling: complete graph and lifecycle fixtures passed; starting real host checks");
  await page.setViewportSize({ width: 960, height: 540 });
  const hosts = await hostChecks();
  await writeFile(`${output}/hosts.json`, JSON.stringify(hosts, null, 2));
  console.log("bake-scheduling: host cache and preparation checks passed; starting live tile-edge reload");
  const live = await liveEdges();
  const results = { graph, hosts, live };
  assert.deepEqual(errors, [], `page errors: ${errors.join("\n")}`);
  await writeFile(`${output}/report.json`, JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ ok: true, output, results }, null, 2));
} catch (error) {
  if (error?.code === "ERR_ASSERTION" || String(error).includes("ASSERT:")) process.exitCode = 3;
  else process.exitCode = 1;
  console.error(error);
} finally { await browser.close(); }
