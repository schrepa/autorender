/** Bounded browser collision baseline. Routed fixtures never modify authored source.
 * Run after a development build: node web/scripts/collision-benchmark.mjs
 * End-to-end query timings include the host, JSON, identity mapping and cache scans.
 */
import assert from "node:assert/strict";
import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { arch, cpus, platform, release } from "node:os";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const workspace = fileURLToPath(new URL("../../", import.meta.url));
const options = { url: process.env.DEMO_URL ?? "http://127.0.0.1:5173/", repeats: 3, warmup: 30, samples: 120, query_samples: 100, query_warmup: 20 };
const flags = new Map([["--url", "url"], ["--output", "output"], ["--repeats", "repeats"], ["--warmup", "warmup"], ["--samples", "samples"], ["--query-samples", "query_samples"], ["--query-warmup", "query_warmup"]]);
const seen = new Set();
for (let index = 2; index < process.argv.length; index += 2) {
  const flag = process.argv[index], key = flags.get(flag), value = process.argv[index + 1];
  assert.ok(key && value && !value.startsWith("--") && !seen.has(flag), "Expected unique supported flags with values.");
  seen.add(flag); options[key] = value;
}
for (const [key, min, max] of [["repeats", 1, 5], ["warmup", 0, 120], ["samples", 1, 240], ["query_samples", 1, 200], ["query_warmup", 0, 100]]) {
  assert.match(String(options[key]), /^[0-9]+$/, key);
  options[key] = Number(options[key]);
  assert.ok(Number.isInteger(options[key]) && options[key] >= min && options[key] <= max, `${key} must be ${min}..${max}`);
}
const base = new URL(`${String(options.url).replace(/\/$/, "")}/`);
assert.ok(["http:", "https:"].includes(base.protocol), "URL must use HTTP(S)");
const artifact = resolve(options.output ?? `${workspace}/artifacts/benchmarks/collision-${new Date().toISOString().replaceAll(":", "-")}-${randomUUID()}.json`);
const hash = value => createHash("sha256").update(value).digest("hex");
const viewport = { width: 1200, height: 900 };
const launchFlags = ["--enable-unsafe-webgpu", ...(process.env.SOFTWARE_GPU === "1" ? ["--use-angle=swiftshader"] : [])];
const workloads = [
  { id: "player-only", total_colliders: 1, sensors: false },
  { id: "sparse-static-1000", total_colliders: 1000, sensors: false },
  { id: "sparse-static-10000", total_colliders: 10000, sensors: false },
  { id: "sparse-sensors-1000", total_colliders: 1000, sensors: true },
  { id: "sparse-sensors-10000", total_colliders: 10000, sensors: true },
];

function fixture(workload) {
  const shape = { kind: "cuboid", half_extents: [0.4, 0.4, 0.4] };
  const collider = { shape, sensor: workload.sensors };
  const template = Array.from({ length: 50 }, (_, index) => {
    const slot = index + 1;
    return { id: `c${slot}`, transform: { translation: [(slot % 10) * 4, 0, Math.floor(slot / 10) * 4] }, collider };
  });
  const document = {
    version: 4, id: workload.id, revision: 1,
    camera: { position: [15, 20, 30], target: [0, 0, 0] },
    meshes: [{ id: "box", source: { kind: "cube" } }],
    materials: [{ id: "blue", color: [0.1, 0.6, 0.85, 1] }],
    prefabs: workload.total_colliders > 1 ? [{ id: "cluster", entities: template }] : [],
    entities: [{ id: "player", tags: ["player"], mesh: "box", material: "blue", transform: { translation: [0, 10, 0] }, collider: { shape, memberships: 2, filter: 1 } }],
  };
  let remaining = workload.total_colliders - 1, group = 0;
  while (remaining > 0) {
    const translation = [20 + (group % 14) * 48, 0, Math.floor(group / 14) * 28];
    const entry = { id: `g${group}`, transform: { translation }, collider };
    if (remaining >= 51) { entry.prefab = "cluster"; remaining -= 51; }
    else remaining--;
    document.entities.push(entry); group++;
  }
  const source = JSON.stringify(document);
  assert.ok(Buffer.byteLength(source) <= 1024 * 1024, "Fixture exceeds authored source byte limit");
  return { document, bytes: Buffer.byteLength(source), sha256: hash(source), source };
}
const fixtures = new Map(workloads.map(workload => [workload.id, fixture(workload)]));
const queries = [
  { op: "raycast", origin: [20, 10, 0], direction: [0, -1, 0], max_distance: 20 },
  { op: "overlap", position: [20, 0, 0], shape: { kind: "sphere", radius: 0.2 }, limit: 20 },
  { op: "shape_cast", position: [20, 10, 0], shape: { kind: "sphere", radius: 0.2 }, direction: [0, -1, 0], max_distance: 20 },
];
function distribution(values) {
  assert.ok(values.length && values.every(value => Number.isFinite(value) && value >= 0));
  const sorted = [...values].sort((a, b) => a - b);
  const quantile = fraction => { const index = (sorted.length - 1) * fraction, low = Math.floor(index); return sorted[low] + (sorted[Math.ceil(index)] - sorted[low]) * (index - low); };
  return { min: sorted[0], median: quantile(0.5), p95: quantile(0.95), p99: quantile(0.99), max: sorted.at(-1) };
}
async function deadline(promise, ms, label) {
  let timer;
  try { return await Promise.race([promise, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${label} exceeded ${ms} ms; no command retried`)), ms); })]); }
  finally { clearTimeout(timer); }
}
async function command(page, request, timeout = 10_000) {
  const reply = await deadline(page.evaluate(request => window.__engineDev.request(request), request), timeout, request.op);
  assert.equal(reply.ok, true, JSON.stringify(reply).slice(0, 3000));
  return reply;
}
const report = {
  version: 1, kind: "browser_engine_collision_baseline", suite: "sparse-collision-v1", status: "incomplete", started_at: new Date().toISOString(),
  configuration: { ...options, viewport, device_scale_factor: 1, workloads, queries, query_filter: "default symmetric memberships=1/filter=MAX, sensors included", order: "fresh-page-per-run; workload order reverses on odd repeats", fixed_mode: "step", query_deadline_ms: 10_000, fixed_deadline_ms: 10_000, suite_deadline_ms: 600_000 },
  environment: { browser_version: null, browser_channel: process.env.BROWSER_PATH ? "explicit-executable" : process.env.BROWSER_CHANNEL ?? "chrome", executable: process.env.BROWSER_PATH ? resolve(process.env.BROWSER_PATH) : null, headless: true, launch_flags: launchFlags, software_gpu_requested: process.env.SOFTWARE_GPU === "1", platform: platform(), arch: arch(), os_release: release(), cpu_model: cpus()[0]?.model ?? null, logical_cpus: cpus().length },
  source: { build: process.env.ENGINE_BUILD ?? null, wasm: null, host: null, fixtures: Object.fromEntries([...fixtures].map(([id, { source, ...value }]) => [id, value])) },
  methodology: "The minimum playground contains one required player collider; zero additional colliders is one total, not an empty plugin world. Non-player cuboids are stationary and disjoint; sensor fixtures produce zero active pairs. One rendered player falls under playground gravity, so fixed-step CPU includes gameplay/controller, transform boundaries, cache rebuilds, sensor detection and rendering. Query sampling runs paused at tick zero before stepping. End-to-end browser performance.now timings include async host dispatch, JSON, authored identity indexing, authoritative transform/cache full scans and response assembly; they are not collision-kernel timings. Frame CPU excludes JSON/stat collection and never measures GPU execution. This sparse baseline does not cover dense overlaps, worst-case query budgets or rigid-body physics. Compare only identical workload, artifacts, browser and hardware metadata; examine repeat variation. No improvement claim is produced.",
  runs: [],
};
let browser, activePage, suiteTimer;
async function measure(workload, repeat, order) {
  const page = await browser.newPage({ viewport, deviceScaleFactor: 1 }); activePage = page;
  page.setDefaultTimeout(20_000);
  const errors = [], wasm = [], scripts = [];
  page.on("pageerror", error => { if (errors.length < 8) errors.push(String(error).slice(0, 1000)); });
  page.on("console", entry => { if (entry.type() === "error" && errors.length < 8) errors.push(entry.text().slice(0, 1000)); });
  page.on("response", response => {
    if (new URL(response.url()).pathname.endsWith(".wasm") && wasm.length < 2) wasm.push(response);
    if (response.request().resourceType() === "script" && scripts.length < 65) scripts.push(response);
  });
  try {
    await page.route("**/scenes/playground.scene.json*", route => route.fulfill({ contentType: "application/json", body: fixtures.get(workload.id).source }));
    const url = new URL("playground.html", base); url.searchParams.set("engine_dev", "1");
    if (process.env.ENGINE_BUILD) url.searchParams.set("engine_build", process.env.ENGINE_BUILD);
    await page.goto(url.href, { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => window.__engineDev?.version === 1);
    const initial = await command(page, { op: "pause" });
    assert.equal(initial.observation.tick, "0"); assert.equal(initial.result.paused, true);
    assert.equal(initial.observation.scene.id, workload.id);
    assert.equal(wasm.length, 1); assert.ok(scripts.length > 0 && scripts.length <= 64);
    const loaded = [];
    for (const response of [...wasm, ...scripts]) {
      const maximum = wasm.includes(response) ? 32 * 1024 * 1024 : 1024 * 1024;
      assert.ok(Number(response.headers()["content-length"] ?? 0) <= maximum);
      const bytes = await response.body(); assert.ok(bytes.length <= maximum);
      loaded.push({ pathname: new URL(response.url()).pathname, bytes: bytes.length, sha256: hash(bytes) });
    }
    const loadedWasm = loaded.shift();
    loaded.sort((a, b) => a.pathname.localeCompare(b.pathname));
    assert.ok(loaded.reduce((sum, module) => sum + module.bytes, 0) <= 8 * 1024 * 1024);
    const host = { sha256: hash(JSON.stringify(loaded)), modules: loaded };
    if (report.source.wasm) assert.deepEqual(loadedWasm, report.source.wasm, "Wasm changed during suite"); else report.source.wasm = loadedWasm;
    if (report.source.host) assert.deepEqual(host, report.source.host, "Host modules changed during suite"); else report.source.host = host;
    const identity = await page.evaluate(async () => {
      const info = (await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" }))?.info;
      return { instance: window.__engineDev.instance, build: window.__engineDev.build, user_agent: navigator.userAgent, hardware_concurrency: navigator.hardwareConcurrency, adapter: info ? { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description, is_fallback_adapter: info.isFallbackAdapter ?? null } : null };
    });
    await command(page, { op: "input" });
    await command(page, { op: "render" });
    const cache = (await command(page, { op: "inspect", component: "CollisionWorld" })).result.value;
    assert.equal(cache.complete, true); assert.equal(cache.colliders, workload.total_colliders); assert.equal(cache.active_colliders, workload.total_colliders);
    const queryResults = await deadline(page.evaluate(async ({ queries, warmup, samples, expectedCount }) => {
      const api = window.__engineDev, before = await api.request({ op: "status" }), started = performance.now(), results = [];
      const check = reply => { if (!reply.ok) throw new Error(JSON.stringify(reply).slice(0, 2000)); return reply; };
      check(before);
      for (const query of queries) {
        const measured = []; let first, last;
        for (let index = 0; index < warmup + samples; index++) {
          if (performance.now() - started > 10_000) throw new Error("Query batch exceeded ten seconds");
          const start = performance.now(), reply = await api.request(query), elapsed = performance.now() - start;
          check(reply);
          const value = reply.result;
          if (value.complete !== true || value.snapshot.complete !== true || value.snapshot.active_colliders !== expectedCount) throw new Error("Incomplete or changed query snapshot");
          const ids = query.op === "overlap" ? value.hits.map(hit => hit.id) : value.hit ? [value.hit.entity.id] : [];
          if (JSON.stringify(ids) !== JSON.stringify(expectedCount === 1 ? [] : ["g0"])) throw new Error(`Unexpected query hits: ${JSON.stringify(ids)}`);
          if (index >= warmup) { measured.push(elapsed); first ??= value.snapshot; last = value.snapshot; }
        }
        results.push({ op: query.op, samples_ms: measured, first_snapshot: first, last_snapshot: last });
      }
      const after = check(await api.request({ op: "status" }));
      if (JSON.stringify(before.observation) !== JSON.stringify(after.observation)) throw new Error("Queries changed authoritative clocks or prepared presentation");
      return { elapsed_ms: performance.now() - started, observation: before.observation, results };
    }, { queries, warmup: options.query_warmup, samples: options.query_samples, expectedCount: workload.total_colliders }), 12_000, "query batch");
    for (const measurement of queryResults.results) measurement.end_to_end_ms = distribution(measurement.samples_ms);
    const benchmark = (await command(page, { op: "benchmark", mode: "step", warmup: options.warmup, samples: options.samples }, 12_000)).result.benchmark;
    assert.equal(benchmark.observation.start.tick, "0");
    assert.equal(benchmark.observation.end.tick, String(options.warmup + options.samples));
    for (const key of ["missing_assets", "invalid_transforms", "dropped_instances"]) assert.equal(benchmark.rendering.ranges[key].max, 0, key);
    assert.equal(benchmark.rendering.ranges.render_candidates.min, 1);
    assert.equal(benchmark.rendering.ranges.render_candidates.max, 1);
    const finalCache = (await command(page, { op: "inspect", component: "CollisionWorld" })).result.value;
    assert.equal(finalCache.complete, true); assert.equal(finalCache.active_colliders, workload.total_colliders);
    const progress = (await command(page, { op: "inspect", component: "GameProgress" })).result.value;
    assert.equal(progress.collision_errors, 0);
    const eventResult = (await command(page, { op: "collision_events" })).result;
    const events = { tick: eventResult.tick, committed_tick: eventResult.committed_tick, complete: eventResult.complete, active_pairs: eventResult.active_pairs, candidate_pairs: eventResult.candidate_pairs, total: eventResult.total, error: eventResult.error };
    assert.equal(events.complete, true); assert.equal(events.active_pairs, 0); assert.equal(events.total, 0);
    assert.deepEqual(errors, []);
    return { workload: workload.id, repeat, order, total_colliders: workload.total_colliders, additional_colliders: workload.total_colliders - 1, sensor_colliders: workload.sensors ? workload.total_colliders - 1 : 0, identity, cache_before: cache, queries: queryResults, benchmark, cache_after: finalCache, progress, events };
  } finally { activePage = undefined; await page.close(); }
}
try {
  for (const [path, key] of [["Cargo.lock", "cargo_lock_sha256"], ["web/package-lock.json", "package_lock_sha256"]]) report.source[key] = hash(await readFile(resolve(workspace, path)));
  browser = await chromium.launch({ ...(process.env.BROWSER_PATH ? { executablePath: resolve(process.env.BROWSER_PATH) } : { channel: process.env.BROWSER_CHANNEL ?? "chrome" }), headless: true, args: launchFlags, timeout: 30_000 });
  report.environment.browser_version = browser.version();
  suiteTimer = setTimeout(() => { void browser.close().catch(() => {}); }, 600_000);
  let order = 0;
  for (let repeat = 0; repeat < options.repeats; repeat++) for (const workload of repeat % 2 ? [...workloads].reverse() : workloads) {
    process.stderr.write(`${JSON.stringify({ event: "collision_benchmark_run", workload: workload.id, repeat, run: order + 1, total: workloads.length * options.repeats })}\n`);
    report.runs.push(await measure(workload, repeat, order++));
  }
  report.status = "complete";
  report.summary = workloads.map(workload => {
    const runs = report.runs.filter(run => run.workload === workload.id);
    return { workload: workload.id, repeats: runs.length, simulation_median_ms_by_run: runs.map(run => run.benchmark.cpu_ms.simulation.median), frame_p95_ms_by_run: runs.map(run => run.benchmark.frame_interval_ms.p95), queries: queries.map(query => ({ op: query.op, median_ms_by_run: runs.map(run => run.queries.results.find(result => result.op === query.op).end_to_end_ms.median), p95_ms_by_run: runs.map(run => run.queries.results.find(result => result.op === query.op).end_to_end_ms.p95) })) };
  });
} catch (error) {
  report.error = { message: String(error.message ?? error).slice(0, 4000) }; process.exitCode = 1;
} finally {
  clearTimeout(suiteTimer);
  await activePage?.close().catch(() => {}); await browser?.close().catch(() => {});
  report.completed_at = new Date().toISOString();
  const bytes = JSON.stringify(report, null, 2); assert.ok(Buffer.byteLength(bytes) <= 2 * 1024 * 1024, "Report exceeded 2 MiB");
  await mkdir(dirname(artifact), { recursive: true }); await writeFile(artifact, `${bytes}\n`, { flag: "wx" });
  console.log(JSON.stringify({ ok: report.status === "complete", artifact, completed_runs: report.runs.length, error: report.error ?? null, summary: report.summary ?? null }));
}
