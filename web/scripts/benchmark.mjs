/** Repeatable, bounded A/B browser measurements through the optional dev API. */
import { createHash, randomUUID } from "node:crypto";
import { mkdir, open, readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { browserChannel, launchFlags, launchOptions } from "./browser-launch.mjs";
import { developmentReady } from "./harness.mjs";
import { arch, cpus, platform, release, totalmem } from "node:os";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import { setTimeout as delay } from "node:timers/promises";
import {
  REPORT_VERSION, SUITE, MAX_REPORT_BYTES, parseArguments, failure,
  validateReport, validateMeasurement, summarizeRuns, compareReports, compareCulling,
  parseSceneIdentity,
} from "./benchmark-report.mjs";

const workspace = fileURLToPath(new URL("../../", import.meta.url));
const viewport = { width: 1440, height: 1000 };
const camera = (position, target, fov = Math.PI / 3, far = 1000) => ({ position, target, vertical_fov_radians: fov, near: 0.1, far });
const workloads = [
  { id: "orbit-1000-overview", game: "orbit", cubes: 1000, visibility: "all", camera: camera([0, 75, 110], [0, 0, 0]) },
  { id: "orbit-10000-partial", game: "orbit", cubes: 10000, visibility: "partial", camera: camera([0, 31, 64], [0, 0, 0]) },
  { id: "orbit-50000-partial", game: "orbit", cubes: 50000, visibility: "partial", camera: camera([0, 31, 64], [0, 0, 0]) },
  { id: "orbit-10000-away", game: "orbit", cubes: 10000, visibility: "none", camera: camera([0, 31, 64], [0, 70, 140]) },
  { id: "playground-overview", game: "playground", cubes: null, visibility: "all", camera: camera([17, 21, 26], [0, 0, 0], 48 * Math.PI / 180, 150) },
];
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
let report, artifact, browser, activePage, suiteTimer, suiteExpired = false, interrupted;
const interrupt = signal => {
  interrupted = signal;
  void browser?.close().catch(() => {});
};
const onInterrupt = () => interrupt("SIGINT"), onTerminate = () => interrupt("SIGTERM");
process.on("SIGINT", onInterrupt);
process.on("SIGTERM", onTerminate);

function expect(condition, code, path, message) { if (!condition) throw failure(code, path, message); }

async function boundedJson(path) {
  const file = await open(path, "r");
  try {
    expect((await file.stat()).size <= MAX_REPORT_BYTES, "report_too_large", path, "Baseline exceeds 2 MiB.");
    const bytes = Buffer.alloc(MAX_REPORT_BYTES + 1);
    let length = 0;
    while (length < bytes.length) {
      const { bytesRead } = await file.read(bytes, length, bytes.length - length, null);
      if (!bytesRead) break;
      length += bytesRead;
    }
    expect(length <= MAX_REPORT_BYTES, "report_too_large", path, "Baseline exceeds 2 MiB.");
    try { return JSON.parse(bytes.subarray(0, length).toString("utf8")); }
    catch { throw failure("invalid_baseline", path, "Baseline is not valid JSON."); }
  } finally { await file.close(); }
}

async function persist() {
  const bytes = Buffer.from(`${JSON.stringify(report, null, 2)}\n`);
  expect(bytes.length <= MAX_REPORT_BYTES, "report_too_large", artifact, "Benchmark report exceeds 2 MiB.");
  await mkdir(dirname(artifact), { recursive: true });
  const file = await open(artifact, "wx");
  try { await file.writeFile(bytes); await file.sync(); }
  finally { await file.close(); }
}

async function sceneSnapshot(base) {
  const response = await fetch(new URL("scenes/playground.scene.json", base), { signal: AbortSignal.timeout(15_000) });
  expect(response.ok, "source_unavailable", "$.source.scene", `Scene fetch failed with HTTP ${response.status}.`);
  const maximum = 1024 * 1024;
  expect(Number(response.headers.get("content-length") ?? 0) <= maximum, "source_too_large", "$.source.scene", "Authored scene exceeds 1 MiB.");
  const reader = response.body.getReader(), chunks = [];
  let length = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > maximum) { await reader.cancel(); throw failure("source_too_large", "$.source.scene", "Authored scene exceeds 1 MiB."); }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  return Buffer.concat(chunks, length);
}

async function deadline(promise, milliseconds, label) {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(failure("benchmark_command_timeout", "$", `${label} exceeded its bounded wait; no mutation was retried.`)), milliseconds);
    })]);
  } finally { clearTimeout(timer); }
}

async function command(page, value, milliseconds = 5000) {
  const reply = await deadline(page.evaluate(command => window.__engineDev.request(command), value), milliseconds, value.op);
  if (!reply.ok) throw failure(reply.error?.code ?? "browser_command_failed", reply.error?.path ?? "$", reply.error?.message ?? "Development command failed.");
  return reply;
}

function near(actual, expected, path, epsilon = 0.0001) {
  expect(Array.isArray(actual) && actual.length === expected.length && actual.every((value, index) => Number.isFinite(value) && Math.abs(value - expected[index]) <= epsilon), "workload_mismatch", path, "Observed camera differs from the configured camera.");
}

function direction(camera) {
  const vector = camera.target.map((value, index) => value - camera.position[index]);
  const magnitude = Math.hypot(...vector);
  return vector.map(value => value / magnitude);
}

function checkCamera(actual, expected) {
  near(actual.position, expected.position, "$.camera.position");
  near(direction(actual), direction(expected), "$.camera.direction");
  near([actual.vertical_fov_radians, actual.near, actual.far], [expected.vertical_fov_radians, expected.near, expected.far], "$.camera.lens");
}

function checkRendering(benchmark, workload, enabled) {
  const ranges = benchmark.rendering.ranges;
  if (workload.cubes !== null) expect(ranges.render_candidates.min === workload.cubes && ranges.render_candidates.max === workload.cubes, "workload_mismatch", "$.rendering.ranges.render_candidates", "Orbit count changed during measured frames.");
  if (!enabled || workload.visibility === "all") expect(ranges.culled_instances.max === 0, "workload_mismatch", "$.rendering.ranges.culled_instances", "All-visible/disabled workloads must remain uncullled throughout measured frames.");
  else if (workload.visibility === "none") expect(ranges.visible_instances.max === 0 && ranges.draw_calls.max === 0, "workload_mismatch", "$.rendering.ranges", "Away workload submitted work during measured frames.");
  else expect(ranges.visible_instances.min > 0 && ranges.culled_instances.min > 0, "workload_mismatch", "$.rendering.ranges", "Partial workloads must retain visible and culled candidates throughout measurement.");
  for (const value of [benchmark.rendering.first, benchmark.rendering.last]) {
    expect(value.culling_enabled === enabled, "workload_mismatch", "$.rendering.culling_enabled", "The requested culling mode was not applied.");
    expect(value.invalid_transforms === 0 && value.missing_assets === 0, "invalid_workload", "$.rendering", "Invalid transforms or missing assets invalidate a benchmark.");
    if (workload.cubes !== null) expect(value.render_candidates === workload.cubes, "workload_mismatch", "$.rendering.render_candidates", "Orbit candidate count differs from the configured workload.");
    expect(value.visible_instances + value.culled_instances === value.render_candidates && value.instances === value.visible_instances, "workload_mismatch", "$.rendering", "Visible/culled/submitted counters do not account for all valid candidates.");
    if (!enabled || workload.visibility === "all") expect(value.culled_instances === 0, "workload_mismatch", "$.rendering.culled_instances", "An all-visible or disabled-culling workload culled instances.");
    else if (workload.visibility === "none") expect(value.visible_instances === 0 && value.draw_calls === 0, "workload_mismatch", "$.rendering", "The away camera must submit no instances or draws with culling enabled.");
    else expect(value.visible_instances > 0 && value.culled_instances > 0, "workload_mismatch", "$.rendering", "The partial workload must include visible and culled instances.");
  }
}

async function measure(options, workload, repeat, enabled, order, sceneBytes, baselineEnvironment) {
  const page = await browser.newPage({ viewport, deviceScaleFactor: 1 });
  activePage = page;
  const errors = [], wasm = [], scripts = [];
  page.setDefaultTimeout(20_000);
  page.on("pageerror", error => { if (errors.length < 8) errors.push(String(error).slice(0, 1000)); });
  page.on("console", entry => { if (entry.type() === "error" && errors.length < 8) errors.push(entry.text().slice(0, 1000)); });
  page.on("response", response => {
    if (new URL(response.url()).pathname.endsWith(".wasm")) wasm.push(response);
    if (response.request().resourceType() === "script") scripts.push(response);
  });
  try {
    const url = new URL(workload.game === "orbit" ? "" : "playground.html", options.url);
    url.searchParams.set("engine_dev", "1");
    if (workload.game === "playground") await page.route("**/scenes/playground.scene.json", route => route.fulfill({ body: sceneBytes, contentType: "application/json" }));
    await page.goto(url.href, { waitUntil: "domcontentloaded", timeout: 20_000 });
    await developmentReady(page, { timeout: 20_000 });
    const pauseDeadline = Date.now() + 5000;
    while (true) {
      const reply = await page.evaluate(() => window.__engineDev.request({ op: "pause" }));
      if (reply.error?.code === "command_busy" && Date.now() < pauseDeadline) { await delay(25); continue; }
      if (!reply.ok) throw failure(reply.error?.code ?? "startup_failed", "$", reply.error?.message ?? "Cannot pause benchmark page.");
      expect(reply.observation.tick === "0" && reply.result.paused, "workload_mismatch", "$.observation.tick", "A fresh development page must be paused at tick zero.");
      break;
    }
    expect(wasm.length === 1, "artifact_unavailable", "$.source.wasm", "Expected exactly one loaded Wasm artifact per game page.");
    expect(Number(wasm[0].headers()["content-length"] ?? 0) <= 32 * 1024 * 1024, "artifact_too_large", "$.source.wasm", "Wasm benchmark artifacts must fit 32 MiB.");
    const wasmBytes = await wasm[0].body();
    expect(wasmBytes.length <= 32 * 1024 * 1024, "artifact_too_large", "$.source.wasm", "Wasm benchmark artifacts must fit 32 MiB.");
    const loadedHash = hash(wasmBytes);
    const known = report.source.wasm[workload.game];
    expect(!known || known.sha256 === loadedHash, "artifact_changed", "$.source.wasm", "Wasm changed during the suite; stop rebuilding while benchmarking.");
    report.source.wasm[workload.game] = { sha256: loadedHash, bytes: wasmBytes.length, pathname: new URL(wasm[0].url()).pathname };
    if (!report.environment.browser_probe) {
      report.environment.browser_probe = await page.evaluate(async () => {
        const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
        const info = adapter?.info;
        return { user_agent: navigator.userAgent, hardware_concurrency: navigator.hardwareConcurrency,
          adapter: info ? { vendor: info.vendor, architecture: info.architecture, device: info.device,
            description: info.description, is_fallback_adapter: info.isFallbackAdapter ?? null } : null };
      });
    }
    if (baselineEnvironment) expect(isDeepStrictEqual(report.environment, baselineEnvironment), "incompatible_baseline", "$.environment", "Browser/version, platform, CPU, adapter probe or launch settings differ from baseline.");
    if (workload.cubes !== null) await page.locator("#count").selectOption(String(workload.cubes));
    await command(page, { op: "input" });
    await command(page, { op: "camera", ...workload.camera });
    await command(page, { op: "culling", enabled });
    if (options.setupTicks) await command(page, { op: "step", ticks: options.setupTicks });
    const before = (await command(page, { op: "inspect", component: "Camera" })).result.value;
    checkCamera(before, workload.camera);
    const scriptCount = scripts.length;
    expect(scriptCount > 0 && scriptCount <= 64, "artifact_count", "$.source.host", "Expected at most 64 loaded host modules per game.");
    const modules = [];
    let hostBytes = 0;
    for (const response of scripts) {
      expect(Number(response.headers()["content-length"] ?? 0) <= 1024 * 1024, "artifact_too_large", "$.source.host", "One host module exceeds 1 MiB.");
      const bytes = await response.body();
      hostBytes += bytes.length;
      expect(bytes.length <= 1024 * 1024 && hostBytes <= 8 * 1024 * 1024, "artifact_too_large", "$.source.host", "Host modules exceed the 1 MiB individual / 8 MiB aggregate collection budget.");
      modules.push({ pathname: new URL(response.url()).pathname, sha256: hash(bytes), bytes: bytes.length });
    }
    modules.sort((a, b) => a.pathname.localeCompare(b.pathname) || a.sha256.localeCompare(b.sha256));
    const host = { sha256: hash(JSON.stringify(modules)), modules };
    const previousHost = report.source.host[workload.game];
    expect(!previousHost || previousHost.sha256 === host.sha256, "artifact_changed", "$.source.host", "Loaded host modules changed during the suite; stop host edits while benchmarking.");
    report.source.host[workload.game] = host;
    const reply = await command(page, { op: "benchmark", warmup: options.warmup, samples: options.samples, mode: options.mode }, 15_000);
    expect(scripts.length === scriptCount, "artifact_changed", "$.source.host", "Additional host code loaded during measurement; the run does not match its recorded module snapshot.");
    const benchmark = reply.result.benchmark;
    validateMeasurement(benchmark, report.configuration);
    checkRendering(benchmark, workload, enabled);
    const after = (await command(page, { op: "inspect", component: "Camera" })).result.value;
    checkCamera(after, workload.camera);
    expect(errors.length === 0, "browser_diagnostics", "$", `Browser errors invalidate measurement: ${errors.join(" | ")}`);
    return { workload: workload.id, repeat, culling: enabled, order, camera: { before, after }, benchmark };
  } finally {
    activePage = undefined;
    await page.close();
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  artifact = resolve(options.output ?? `${workspace}/artifacts/benchmarks/${new Date().toISOString().replaceAll(":", "-")}-${randomUUID()}.json`);
  let baseline;
  if (options.baseline) { baseline = await boundedJson(resolve(options.baseline)); validateReport(baseline); }
  const configuration = { mode: options.mode, warmup: options.warmup, samples: options.samples, repeats: options.repeats,
    viewport, device_scale_factor: 1, workloads, scheduling: "fresh-page-per-run-alternating-arm-order-v1" };
  if (options.setupTicks) configuration.setup_ticks = options.setupTicks;
  if (options.mode === "present") configuration.presentation_alpha = 0.5;
  if (baseline) expect(isDeepStrictEqual(configuration, baseline.configuration), "incompatible_baseline", "$.configuration", "Baseline configuration differs; use identical workload/sample settings.");
  report = { version: REPORT_VERSION, kind: "browser_engine_benchmark", suite: SUITE, status: "incomplete", started_at: new Date().toISOString(),
    configuration, environment: { browser_channel: browserChannel, browser_version: null, platform: platform(), arch: arch(), os_release: release(),
      cpu_model: cpus()[0]?.model ?? null, logical_cpus: cpus().length, physical_memory_bytes: totalmem(), headless: true,
      launch_flags: launchFlags, software_gpu_requested: process.env.SOFTWARE_GPU === "1", browser_probe: null },
    source: { wasm: {}, host: {}, scene: null, cargo_lock_sha256: null, package_lock_sha256: null }, runs: [],
    methodology: "Fresh paused development pages at tick zero, followed by configured exact setup ticks; fixed cameras and viewport; alternating culling order. Present samples use alpha .5 without advancing simulation. Per-run RAF-paced CPU/frame distributions exclude warm-up and setup. Summary is the median of each run statistic, not pooled percentiles. CPU submission is not GPU execution. Asset byte counts exclude process/JS/Wasm heaps and driver allocations; presentation history counters describe their own storage only.",
  };
  for (const [file, field] of [["Cargo.lock", "cargo_lock_sha256"], ["web/package-lock.json", "package_lock_sha256"]]) {
    try { report.source[field] = hash(await readFile(resolve(workspace, file))); } catch { /* A remote/local-server bundle may have no adjacent checkout metadata. */ }
  }
  const { chromium } = await import("@playwright/test");
  browser = await chromium.launch(launchOptions({ timeout: 30_000, handleSIGINT: false, handleSIGTERM: false }));
  report.environment.browser_version = browser.version();
  suiteTimer = setTimeout(() => {
    suiteExpired = true;
    void browser.close().catch(() => {});
  }, 10 * 60 * 1000);
  const sceneBytes = await sceneSnapshot(options.url);
  report.source.scene = { sha256: hash(sceneBytes), bytes: sceneBytes.length, ...parseSceneIdentity(sceneBytes.toString("utf8")) };
  if (baseline) expect(isDeepStrictEqual(report.source.scene, baseline.source.scene), "incompatible_baseline", "$.source.scene", "Authored scene snapshot differs from baseline.");
  let order = 0;
  for (let repeat = 0; repeat < options.repeats; repeat++) for (const workload of workloads) {
    for (const enabled of repeat % 2 === 0 ? [false, true] : [true, false]) {
      if (suiteExpired) throw failure("benchmark_suite_timeout", "$", "The complete benchmark suite exceeded ten minutes.");
      if (interrupted) throw failure("benchmark_interrupted", "$", `Benchmark interrupted by ${interrupted}.`);
      process.stderr.write(`${JSON.stringify({ event: "benchmark_run", workload: workload.id, repeat, culling: enabled, run: order + 1, total: workloads.length * options.repeats * 2 })}\n`);
      report.runs.push(await measure(options, workload, repeat, enabled, order++, sceneBytes, baseline?.environment));
    }
  }
  report.status = "complete";
  report.completed_at = new Date().toISOString();
  report.summary = summarizeRuns(report.runs, configuration);
  report.culling_comparison = compareCulling(report.summary);
  if (baseline) report.baseline_comparison = compareReports(report, baseline, options.maxRegressionPercent);
  await persist();
  const passed = report.baseline_comparison?.passed !== false;
  return { version: 1, ok: passed, result: { artifact, status: report.status, runs: report.runs.length, browser: report.environment.browser_version,
    summary: report.summary, culling_comparison: report.culling_comparison, baseline_comparison: report.baseline_comparison ?? null },
    ...(passed ? {} : { error: { code: "benchmark_regression", path: "$.baseline_comparison", message: "One or more configured CPU/frame/asset metrics exceeded the regression threshold." } }) };
}

let response;
try { response = await main(); }
catch (cause) {
  const error = { code: suiteExpired ? "benchmark_suite_timeout" : interrupted ? "benchmark_interrupted" : cause.code ?? "benchmark_failed", path: cause.path ?? "$", message: interrupted ? `Benchmark interrupted by ${interrupted}; owned pages are closing.` : String(cause.message ?? cause).slice(0, 4000) };
  if (report) {
    report.error = error;
    report.completed_at = new Date().toISOString();
    try { await persist(); } catch (writeError) { error.artifact_error = String(writeError.message ?? writeError).slice(0, 1000); artifact = null; }
  }
  response = { version: 1, ok: false, error, result: { artifact: report ? artifact : null, completed_runs: report?.runs.length ?? 0 } };
} finally {
  clearTimeout(suiteTimer);
  await activePage?.close().catch(() => {});
  await browser?.close().catch(() => {});
  process.removeListener("SIGINT", onInterrupt);
  process.removeListener("SIGTERM", onTerminate);
}
process.stdout.write(`${JSON.stringify(response)}\n`);
if (!response.ok) process.exitCode = 1;
