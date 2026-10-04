import assert from "node:assert/strict";
import test from "node:test";
import { parseArguments, parseSceneIdentity, RENDER_FIELDS, statistics, summarizeRuns, compareReports, compareCulling, difference, validateReport } from "./benchmark-report.mjs";

function fixture() {
  const configuration = { mode: "step", warmup: 1, samples: 3, repeats: 2, viewport: { width: 1440, height: 1000 }, device_scale_factor: 1,
    workloads: Array.from({ length: 5 }, (_, index) => ({ id: `workload-${index}`, camera: { position: [0, 1, 2], target: [0, 0, 0] } })) };
  const runs = [];
  const distribution = base => ({ min: base / 2, median: base, p95: base * 2, p99: base * 3, max: base * 4 });
  for (const workload of configuration.workloads) for (const culling of [false, true]) for (let repeat = 0; repeat < configuration.repeats; repeat++) {
    const rendering = { culling_enabled: culling, render_candidates: 1000, visible_instances: culling ? 500 : 1000, culled_instances: culling ? 500 : 0,
      instances: culling ? 500 : 1000, draw_calls: 1, upload_bytes: culling ? 48000 : 96000, resident_meshes: 1, resident_mesh_bytes: 1000,
      resident_textures: 0, resident_texture_bytes: 0, mesh_upload_bytes: 0, texture_upload_bytes: 0, fixed_steps: 1,
      dropped_seconds: 0, dropped_instances: 0, missing_assets: 0, invalid_transforms: 0, simulation_ms: 1, extraction_ms: 1, render_ms: 1 };
    const assets = { meshes: 1, materials: 1, textures: 0, resident_cpu_bytes: 2000, texture_cpu_bytes: 0 };
    runs.push({ workload: workload.id, culling, repeat, camera: workload.camera, benchmark: {
      version: 1, mode: "step", warmup: 1, samples: 3, elapsed_ms: 60,
      frame_interval_ms: distribution(16), cpu_ms: { simulation: distribution(1), extraction: distribution(1), render: distribution(1), total: distribution(3 + repeat * 2) },
      rendering: { first: { ...rendering }, last: { ...rendering }, ranges: Object.fromEntries(RENDER_FIELDS.map(field => [field, { min: rendering[field], max: rendering[field] }])) }, assets: { before: { ...assets }, after: { ...assets } },
      observation: { start: { tick: "0" }, end: { tick: "4" } }, environment: { canvas_width: 1342, canvas_height: 518, device_pixel_ratio: 1, user_agent: "fixture" },
    } });
  }
  return { version: 1, kind: "browser_engine_benchmark", suite: "browser-engine-culling-v1", status: "complete", configuration,
    environment: { browser_version: "153", platform: "test", adapter: "fixture" },
    source: { wasm: { orbit: { sha256: "old-build" } }, scene: { sha256: "scene" } }, runs };
}

test("arguments bound work and require explicit baseline thresholds", () => {
  assert.deepEqual(parseArguments([]), { url: "http://127.0.0.1:5173/", warmup: 30, samples: 120, repeats: 3, mode: "step" });
  const options = parseArguments(["--samples", "1", "--warmup", "0", "--repeats", "5", "--mode", "render", "--baseline", "old.json", "--max-regression-percent", "12.5"]);
  assert.equal(options.maxRegressionPercent, 12.5);
  for (const args of [["--samples", "0"], ["--samples", "241"], ["--warmup", "121"], ["--repeats", "6"], ["--repeats", "1.5"], ["--mode", "unknown"], ["--samples"], ["--output", ""], ["--samples", "2", "--samples", "3"], ["--max-regression-percent", "5"], ["--url", "https://example.com"], ["--url", "http://user:password@localhost"], ["--url", "http://localhost/?secret=value"]]) {
    assert.throws(() => parseArguments(args), error => error.code === "invalid_arguments", JSON.stringify(args));
  }
});

test("statistics interpolate explicit small sample percentiles without mutating input", () => {
  const values = [100, 1, 3, 2];
  const result = statistics(values);
  assert.equal(result.median, 2.5);
  assert.ok(Math.abs(result.p95 - 85.45) < 1e-10);
  assert.deepEqual(statistics([7]), { min: 7, median: 7, p95: 7, p99: 7, max: 7 });
  assert.deepEqual(values, [100, 1, 3, 2]);
  for (const invalid of [[], [NaN], [Infinity], [-1]]) assert.throws(() => statistics(invalid));
});

test("repeat summaries use medians of run statistics and report repeat spread", () => {
  const report = fixture();
  const summary = summarizeRuns(report.runs, report.configuration);
  assert.equal(summary.length, 10);
  assert.equal(summary[0].median_of_runs.total_cpu_median_ms, 4);
  assert.equal(summary[0].median_of_runs.total_cpu_p95_ms, 8);
  assert.equal(summary[0].repeat_spread.total_cpu_median_ms.min, 3);
  assert.equal(summary[0].repeat_spread.total_cpu_median_ms.max, 5);
  const comparison = compareCulling(summary);
  assert.equal(comparison.length, 5);
  assert.equal(comparison[0].enabled_over_disabled.upload_bytes.ratio, 0.5);
  assert.equal(comparison[0].enabled_over_disabled.gpu_texture_bytes.ratio, 1);
});

test("incomplete, duplicate or wrong tick runs cannot become a valid baseline", () => {
  const incomplete = fixture(); incomplete.status = "incomplete";
  assert.throws(() => validateReport(incomplete), error => error.code === "invalid_baseline");
  const missing = fixture(); missing.runs.pop();
  assert.throws(() => validateReport(missing), error => error.code === "incomplete_report");
  const duplicate = fixture(); duplicate.runs[1].repeat = 0;
  assert.throws(() => validateReport(duplicate), error => error.code === "incomplete_report");
  const wrongTick = fixture(); wrongTick.runs[0].benchmark.observation.start.tick = "1";
  assert.throws(() => validateReport(wrongTick), error => error.code === "invalid_report");
  const wrongMode = fixture(); wrongMode.runs[0].benchmark.rendering.last.culling_enabled = true;
  assert.throws(() => validateReport(wrongMode), error => error.code === "invalid_report");
});

test("baseline compatibility rejects browser, workload, device, scene and observed canvas drift", () => {
  const original = fixture();
  for (const change of [
    value => { value.environment.browser_version = "154"; },
    value => { value.environment.adapter = "different-device"; },
    value => { value.configuration.workloads[0].camera.position[0] = 7; },
    value => { value.source.scene.sha256 = "changed-scene"; },
    value => { value.runs[0].benchmark.environment.canvas_width++; },
    value => { value.runs[0].camera = { position: [7, 8, 9] }; },
  ]) {
    const other = structuredClone(original); change(other);
    assert.throws(() => compareReports(other, original), error => error.code === "incompatible_baseline");
  }
});

test("code artifacts may change while comparisons rederive rather than trust saved summaries", () => {
  const original = fixture(), current = fixture();
  current.source.wasm.orbit.sha256 = "new-build";
  current.summary = [{ forged: true }];
  const comparison = compareReports(current, original, 0);
  assert.equal(comparison.passed, true);
  assert.equal(comparison.comparisons[0].metrics.total_cpu_median_ms.ratio, 1);
});

test("regression thresholds cover CPU, cadence and asset memory with explicit zero baselines", () => {
  const original = fixture(), current = fixture();
  for (const run of current.runs) {
    for (const key of ["min", "median", "p95", "p99", "max"]) run.benchmark.cpu_ms.total[key] *= 1.2;
    run.benchmark.rendering.first.resident_texture_bytes = 16;
    run.benchmark.rendering.last.resident_texture_bytes = 16;
    run.benchmark.rendering.ranges.resident_texture_bytes = { min: 16, max: 16 };
  }
  const gated = compareReports(current, original, 10);
  assert.equal(gated.passed, false);
  assert.ok(gated.regressions.some(row => row.metric === "total_cpu_median_ms"));
  assert.ok(gated.regressions.some(row => row.metric === "gpu_texture_bytes" && row.zero_baseline));
  assert.equal(compareReports(current, original).passed, true);
  assert.deepEqual(difference(0, 0), { current: 0, baseline: 0, ratio: 1, percent: 0, zero_baseline: false });
  assert.deepEqual(difference(1, 0), { current: 1, baseline: 0, ratio: null, percent: null, zero_baseline: true });
});

test("middle-frame errors, drops and workload changes invalidate reports even with healthy endpoints", () => {
  for (const field of ["missing_assets", "invalid_transforms", "dropped_instances", "dropped_seconds", "render_candidates"]) {
    const report = fixture();
    report.runs[0].benchmark.rendering.ranges[field].max++;
    assert.throws(() => validateReport(report), error => error.code === "invalid_workload", field);
  }
});

test("source revisions preserve u64 precision and reject unsupported numeric forms", () => {
  for (const revision of ["0", "9007199254740993", "18446744073709551615"]) {
    assert.deepEqual(parseSceneIdentity(`{"id":"scene","version":3,"revision":${revision},"extra":{"revision":5}}`), { id: "scene", version: 3, revision });
  }
  for (const revision of ["18446744073709551616", "1.5", "1e3", '"5"', "-1"]) assert.throws(() => parseSceneIdentity(`{"id":"scene","version":3,"revision":${revision}}`));
});

test("nonfinite and malformed distributions are rejected before report comparison", () => {
  for (const change of [
    value => { value.runs[0].benchmark.cpu_ms.total.p95 = Infinity; },
    value => { value.runs[0].benchmark.cpu_ms.total.median = 1000; },
    value => { value.runs[0].benchmark.assets.after.resident_cpu_bytes = -1; },
    value => { value.runs[0].benchmark.environment.canvas_height = 0; },
  ]) { const report = fixture(); change(report); assert.throws(() => validateReport(report)); }
});

test("fractional presentation requires explicit setup ticks, unchanged simulation and the measured alpha", () => {
  assert.equal(parseArguments(["--mode", "present"]).setupTicks, 1);
  assert.equal(parseArguments(["--mode", "render", "--setup-ticks", "1"]).setupTicks, 1);
  for (const value of ["-1", "121", "1.5", "Infinity"]) assert.throws(() => parseArguments(["--setup-ticks", value]));
  const report = fixture();
  Object.assign(report.configuration, { mode: "present", setup_ticks: 1, presentation_alpha: 0.5 });
  for (const { benchmark } of report.runs) {
    benchmark.mode = "present";
    benchmark.observation = {
      start: { frame: "1", tick: "1", presentation: { frame: "1", tick: "1", mode: "current", alpha: 1, history_nodes: 1000, history_capacity: 1024, history_bytes: 65536, presented_nodes: 1000 } },
      end: { frame: "1", tick: "1", presentation: { frame: "1", tick: "1", mode: "interpolated", alpha: 0.5, history_nodes: 1000, history_capacity: 1024, history_bytes: 65536, presented_nodes: 1000, presentation_scratch_bytes: 32768 } },
    };
  }
  assert.equal(validateReport(report)[0].median_of_runs.history_bytes, 65536);
  assert.equal(validateReport(report)[0].median_of_runs.presentation_scratch_bytes, 32768);
  assert.equal(validateReport(report)[0].median_of_runs.presented_store_bytes, undefined);
  for (const { benchmark } of report.runs) {
    benchmark.observation.end.presentation.presented_store_bytes = 81920;
  }
  assert.equal(validateReport(report)[0].median_of_runs.presented_store_bytes, 81920);
  for (const change of [
    value => { value.runs[0].benchmark.observation.end.tick = "2"; },
    value => { value.runs[0].benchmark.observation.end.presentation.alpha = 1; },
    value => { value.runs[0].benchmark.observation.end.presentation.mode = "current"; },
    value => { value.runs[0].benchmark.observation.end.presentation.tick = "0"; },
    value => { value.runs[0].benchmark.observation.end.presentation.frame = null; },
    value => { value.runs[0].benchmark.observation.end.frame = "2"; },
    value => { value.runs[0].benchmark.observation.end.presentation.history_bytes = -1; },
    value => { value.runs[0].benchmark.observation.end.presentation.presentation_scratch_bytes = -1; },
    value => { value.runs[0].benchmark.observation.end.presentation.presented_store_bytes = -1; },
    value => { value.configuration.setup_ticks = 0; },
    value => { value.configuration.presentation_alpha = 0.2; },
  ]) { const invalid = structuredClone(report); change(invalid); assert.throws(() => validateReport(invalid)); }
  const render = structuredClone(report);
  render.configuration.mode = "render"; delete render.configuration.presentation_alpha;
  for (const { benchmark } of render.runs) {
    benchmark.mode = "render";
    Object.assign(benchmark.observation.end.presentation, { mode: "current", alpha: 1 });
  }
  assert.throws(() => compareReports(report, render), error => error.code === "incompatible_baseline");
});
