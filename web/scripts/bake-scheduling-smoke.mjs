/** Full terrain bake bytes and lifecycle: the shipped coordinator and JobWorker,
 * including Highlands, tile edges, ordered layers, reversed child delivery,
 * malformed results, cancellation and retained coordinator heap capacity.
 */
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { artifacts, launch, openPaused, sceneUrl, routedScene } from "./harness.mjs";

const output = await artifacts("bake-scheduling-smoke");
const { browser, page, errors } = await launch({ width: 320, height: 240, timeout: 600_000 });
const bindingUrls = new Set();
page.on("response", response => {
  if (response.request().resourceType() === "script" && new URL(response.url()).pathname.endsWith("/stress_lab.js")) bindingUrls.add(response.url());
});
const empty = {
  version: 9, id: "bake-input-fixture", revision: 1,
  camera: { position: [0, 20, 40], target: [0, 0, 0], fov_degrees: 60, near: 0.1, far: 2000 },
  materials: [{ id: "plain", color: [0.4, 0.5, 0.6, 1] }],
  meshes: [{ id: "box", source: { kind: "cube" } }],
  entities: [{ id: "player", mesh: "box", material: "plain", tags: ["player"], transform: { translation: [0, 10, 0] } }],
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

try {
  await scene.route();
  const query = { workers: "off", jobcache: "off", effects: "none", ...(process.env.ENGINE_BUILD ? { engine_build: process.env.ENGINE_BUILD } : {}) };
  await openPaused(page, sceneUrl(query), { timeout: 600_000 });
  assert.equal(bindingUrls.size, 1, "one exact loaded game build");
  const moduleUrl = [...bindingUrls][0];
  if (process.env.ENGINE_BUILD) assert.ok(moduleUrl.includes(`/${process.env.ENGINE_BUILD}/`), "the requested build is pinned");
  const results = await page.evaluate(async ({ moduleUrl, fixtures, highlands }) => {
    const gameModule = await import(moduleUrl);
    const workerUrl = moduleUrl.replace(/stress_lab\.js$/, "engine_worker.js");
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
        pools.push({ mode, completed: first.completed.map(value => value.id), failed: first.failed, stats });
      } finally {
        pool.detach();
        for (const slot of pool.slots) slot.worker.terminate();
      }
    }
    const rejects = [];
    const thrown = action => { try { action(); return false; } catch { return true; } };
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
  }, { moduleUrl, fixtures, highlands });
  assert.deepEqual(errors, [], `page errors: ${errors.join("\n")}`);
  await writeFile(`${output}/report.json`, JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ ok: true, output, results }, null, 2));
} catch (error) {
  if (error?.code === "ERR_ASSERTION" || String(error).includes("ASSERT:")) process.exitCode = 3;
  else process.exitCode = 1;
  console.error(error);
} finally { await browser.close(); }
