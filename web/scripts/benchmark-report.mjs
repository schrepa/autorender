/** Pure validation and comparison of bounded browser benchmark reports. */
import { percentiles } from "../src/statistics.ts";

export const REPORT_VERSION = 1;
export const SUITE = "browser-engine-culling-v1";
export const MAX_REPORT_BYTES = 2 * 1024 * 1024;
export const PERCENTILES = ["min", "median", "p95", "p99", "max"];
export const RENDER_FIELDS = ["render_candidates", "visible_instances", "culled_instances", "instances", "dropped_instances", "missing_assets", "invalid_transforms", "draw_calls", "upload_bytes", "fixed_steps", "dropped_seconds", "resident_meshes", "resident_mesh_bytes", "mesh_upload_bytes", "resident_textures", "resident_texture_bytes", "texture_upload_bytes", "simulation_ms", "extraction_ms", "render_ms"];

export function failure(code, path, message) {
  return Object.assign(new Error(message), { code, path });
}

/** Node 24's reviver source context preserves authored u64 revisions exactly. */
export function parseSceneIdentity(source) {
  const revisions = new WeakMap();
  let scene;
  try {
    scene = JSON.parse(source, function (key, value, context) {
      if (key === "revision" && typeof value === "number") revisions.set(this, context?.source);
      return value;
    });
  } catch { throw failure("invalid_scene_snapshot", "$.source.scene", "Authored scene is not valid JSON."); }
  const revision = scene && typeof scene === "object" ? revisions.get(scene) : undefined;
  if (typeof scene?.id !== "string" || !Number.isSafeInteger(scene.version) || typeof revision !== "string" || !/^(0|[1-9][0-9]*)$/.test(revision) || BigInt(revision) > 18_446_744_073_709_551_615n) throw failure("invalid_scene_snapshot", "$.source.scene", "Expected a scene ID/version and exact unsigned 64-bit numeric revision; Node 24 is required.");
  return { id: scene.id, version: scene.version, revision };
}

export function parseArguments(args) {
  const options = { url: "http://127.0.0.1:5173/", warmup: 30, samples: 120, repeats: 3, mode: "step" };
  const seen = new Set();
  const flags = new Map([["--url", "url"], ["--warmup", "warmup"], ["--samples", "samples"], ["--repeats", "repeats"], ["--mode", "mode"], ["--setup-ticks", "setupTicks"], ["--output", "output"], ["--baseline", "baseline"], ["--max-regression-percent", "maxRegressionPercent"]]);
  for (let index = 0; index < args.length; index += 2) {
    const flag = args[index], key = flags.get(flag), value = args[index + 1];
    if (!key || !value || value.startsWith("--") || seen.has(flag)) throw failure("invalid_arguments", flag ?? "$", "Expected unique --url/--warmup/--samples/--repeats/--mode/--setup-ticks/--output/--baseline/--max-regression-percent options with values.");
    seen.add(flag);
    options[key] = value;
  }
  for (const [key, minimum, maximum] of [["warmup", 0, 120], ["samples", 1, 240], ["repeats", 1, 5]]) {
    if (!/^[0-9]+$/.test(String(options[key]))) throw failure("invalid_arguments", `--${key}`, `Expected an integer from ${minimum} through ${maximum}.`);
    options[key] = Number(options[key]);
    if (!Number.isSafeInteger(options[key]) || options[key] < minimum || options[key] > maximum) throw failure("invalid_arguments", `--${key}`, `Expected an integer from ${minimum} through ${maximum}.`);
  }
  if (!["step", "render", "present"].includes(options.mode)) throw failure("invalid_arguments", "--mode", "Mode must be step, render or present.");
  if (options.mode === "present" && options.setupTicks === undefined) options.setupTicks = 1;
  if (options.setupTicks !== undefined) {
    if (!/^[0-9]+$/.test(String(options.setupTicks)) || Number(options.setupTicks) > 120) throw failure("invalid_arguments", "--setup-ticks", "Expected an integer from 0 through 120.");
    options.setupTicks = Number(options.setupTicks);
  }
  if (options.maxRegressionPercent !== undefined) {
    options.maxRegressionPercent = Number(options.maxRegressionPercent);
    if (!Number.isFinite(options.maxRegressionPercent) || options.maxRegressionPercent < 0 || options.maxRegressionPercent > 1000 || !options.baseline) throw failure("invalid_arguments", "--max-regression-percent", "A threshold from 0 through 1000 requires --baseline.");
  }
  let url;
  try { url = new URL(options.url); } catch { throw failure("invalid_arguments", "--url", "Expected a loopback development server URL."); }
  if (url.protocol !== "http:" || !["localhost", "127.0.0.1", "[::1]"].includes(url.hostname) || url.username || url.password || url.search || url.hash) throw failure("invalid_arguments", "--url", "Use an HTTP loopback base URL without credentials, query or fragment.");
  if (!url.pathname.endsWith("/")) url.pathname += "/";
  options.url = url.href;
  return options;
}

/** Percentiles of repeat measurements: their medians and spread. */
export function statistics(values) {
  if (!values.length || values.some(value => !Number.isFinite(value) || value < 0)) throw failure("invalid_measurement", "$", "Measurements must be a nonempty array of finite nonnegative numbers.");
  return percentiles(values);
}

function nonnegative(value, path) {
  if (!Number.isFinite(value) || value < 0) throw failure("invalid_report", path, "Expected a finite nonnegative measurement.");
}

function distribution(value, path) {
  if (!value || typeof value !== "object") throw failure("invalid_report", path, "Expected a complete timing distribution.");
  for (const key of PERCENTILES) nonnegative(value[key], `${path}.${key}`);
  for (let index = 1; index < PERCENTILES.length; index++) if (value[PERCENTILES[index]] < value[PERCENTILES[index - 1]]) throw failure("invalid_report", path, "Timing percentiles must be ordered.");
}

export function validateMeasurement(benchmark, configuration) {
  if (benchmark?.version !== 1 || benchmark.mode !== configuration.mode || benchmark.warmup !== configuration.warmup || benchmark.samples !== configuration.samples) throw failure("invalid_report", "$.benchmark", "Benchmark result does not match the requested complete sample configuration.");
  distribution(benchmark.frame_interval_ms, "$.benchmark.frame_interval_ms");
  for (const phase of ["simulation", "extraction", "render", "total"]) distribution(benchmark.cpu_ms?.[phase], `$.benchmark.cpu_ms.${phase}`);
  nonnegative(benchmark.elapsed_ms, "$.benchmark.elapsed_ms");
  for (const side of ["first", "last"]) {
    const rendering = benchmark.rendering?.[side];
    if (typeof rendering?.culling_enabled !== "boolean") throw failure("invalid_report", `$.benchmark.rendering.${side}.culling_enabled`, "Expected the observed culling mode.");
    for (const key of RENDER_FIELDS) nonnegative(rendering[key], `$.benchmark.rendering.${side}.${key}`);
  }
  for (const key of RENDER_FIELDS) {
    const range = benchmark.rendering.ranges?.[key];
    nonnegative(range?.min, `$.benchmark.rendering.ranges.${key}.min`);
    nonnegative(range?.max, `$.benchmark.rendering.ranges.${key}.max`);
    if (range.min > range.max || [benchmark.rendering.first[key], benchmark.rendering.last[key]].some(value => value < range.min || value > range.max)) throw failure("invalid_report", `$.benchmark.rendering.ranges.${key}`, "Measured ranges must contain first/last values and have ordered bounds.");
  }
  for (const key of ["missing_assets", "invalid_transforms", "dropped_instances", "dropped_seconds"]) {
    if (benchmark.rendering.ranges[key].max !== 0) throw failure("invalid_workload", `$.benchmark.rendering.ranges.${key}`, "A dropped or invalid measured frame cannot be accepted as a faster workload.");
  }
  if (benchmark.rendering.ranges.render_candidates.min !== benchmark.rendering.ranges.render_candidates.max) {
    throw failure("invalid_workload", "$.benchmark.rendering.ranges.render_candidates", "Benchmark candidate count must remain constant throughout measured frames.");
  }
  for (const side of ["before", "after"]) for (const key of ["meshes", "materials", "textures", "resident_cpu_bytes", "texture_cpu_bytes"]) nonnegative(benchmark.assets?.[side]?.[key], `$.benchmark.assets.${side}.${key}`);
  for (const key of ["canvas_width", "canvas_height", "device_pixel_ratio"]) {
    nonnegative(benchmark.environment?.[key], `$.benchmark.environment.${key}`);
    if (benchmark.environment[key] === 0) throw failure("invalid_report", `$.benchmark.environment.${key}`, "A benchmark cannot use a zero-sized canvas or pixel ratio.");
  }
  for (const side of ["start", "end"]) if (!/^(0|[1-9][0-9]*)$/.test(benchmark.observation?.[side]?.tick ?? "")) throw failure("invalid_report", `$.benchmark.observation.${side}.tick`, "Tick must be a decimal string.");
  const start = BigInt(benchmark.observation.start.tick), end = BigInt(benchmark.observation.end.tick);
  const expected = configuration.mode === "step" ? BigInt(configuration.warmup + configuration.samples) : 0n;
  if (start !== BigInt(configuration.setup_ticks ?? 0) || end - start !== expected) throw failure("invalid_report", "$.benchmark.observation", "Every run must start at its configured setup tick and advance only the configured exact steps.");
  for (const side of ["start", "end"]) {
    const presentation = benchmark.observation[side].presentation;
    if (presentation !== undefined) {
      for (const field of ["frame", "tick"]) {
        if (!/^(0|[1-9][0-9]*)$/.test(presentation[field] ?? "") || typeof presentation[field] !== "string") throw failure("invalid_report", `$.benchmark.observation.${side}.presentation.${field}`, "A benchmark must identify its prepared extraction with decimal frame/tick strings.");
        if (benchmark.observation[side][field] !== undefined && presentation[field] !== benchmark.observation[side][field]) throw failure("invalid_report", `$.benchmark.observation.${side}.presentation.${field}`, "Prepared extraction identity must match the observed frame/tick.");
      }
      if (!["current", "interpolated"].includes(presentation.mode) || !Number.isFinite(presentation.alpha) || presentation.alpha < 0 || presentation.alpha > 1 || (presentation.mode === "current" && presentation.alpha !== 1)) throw failure("invalid_report", `$.benchmark.observation.${side}.presentation`, "Expected a valid observed presentation mode and alpha.");
      for (const field of ["history_nodes", "history_capacity", "history_bytes", "presented_nodes"]) {
        if (!Number.isSafeInteger(presentation[field]) || presentation[field] < 0) throw failure("invalid_report", `$.benchmark.observation.${side}.presentation.${field}`, "Expected a nonnegative presentation count.");
      }
      for (const field of ["presentation_scratch_bytes", "presented_store_bytes"]) {
        if (presentation[field] !== undefined && (!Number.isSafeInteger(presentation[field]) || presentation[field] < 0)) throw failure("invalid_report", `$.benchmark.observation.${side}.presentation.${field}`, "Expected a nonnegative presentation allocation.");
      }
    }
  }
  if (configuration.mode === "present" && (configuration.presentation_alpha !== 0.5 || benchmark.observation.end.presentation?.mode !== "interpolated" || benchmark.observation.end.presentation.alpha !== 0.5)) throw failure("invalid_report", "$.benchmark.observation.end.presentation", "Present benchmarks must measure the explicit half-tick pose.");
  if (configuration.mode !== "present" && benchmark.observation.end.presentation !== undefined && benchmark.observation.end.presentation.mode !== "current") throw failure("invalid_report", "$.benchmark.observation.end.presentation", "Exact step/render benchmarks must present the current authoritative pose.");
}

function median(values) { return statistics(values).median; }
function groupedMetrics(runs) {
  const metrics = {};
  for (const phase of ["simulation", "extraction", "render", "total"]) {
    for (const key of PERCENTILES) metrics[`${phase}_cpu_${key}_ms`] = median(runs.map(run => run.benchmark.cpu_ms[phase][key]));
  }
  for (const key of PERCENTILES) metrics[`frame_interval_${key}_ms`] = median(runs.map(run => run.benchmark.frame_interval_ms[key]));
  metrics.cpu_asset_bytes = median(runs.map(run => run.benchmark.assets.after.resident_cpu_bytes));
  metrics.gpu_mesh_bytes = median(runs.map(run => run.benchmark.rendering.last.resident_mesh_bytes));
  metrics.gpu_texture_bytes = median(runs.map(run => run.benchmark.rendering.last.resident_texture_bytes));
  metrics.upload_bytes = median(runs.map(run => run.benchmark.rendering.last.upload_bytes));
  metrics.visible_instances = median(runs.map(run => run.benchmark.rendering.last.visible_instances));
  metrics.culled_instances = median(runs.map(run => run.benchmark.rendering.last.culled_instances));
  metrics.draw_calls = median(runs.map(run => run.benchmark.rendering.last.draw_calls));
  for (const field of ["history_nodes", "history_capacity", "history_bytes", "presented_nodes", "presentation_scratch_bytes", "presented_store_bytes"]) {
    const values = runs.map(run => run.benchmark.observation.end.presentation?.[field]);
    if (values.every(value => value !== undefined)) metrics[field] = median(values);
  }
  return metrics;
}

export function summarizeRuns(runs, configuration) {
  const result = [];
  for (const workload of configuration.workloads) for (const culling of [false, true]) {
    const group = runs.filter(run => run.workload === workload.id && run.culling === culling);
    if (group.length !== configuration.repeats || new Set(group.map(run => run.repeat)).size !== configuration.repeats || group.some(run => !Number.isInteger(run.repeat) || run.repeat < 0 || run.repeat >= configuration.repeats)) throw failure("incomplete_report", "$.runs", `Expected exactly ${configuration.repeats} unique repeats for ${workload.id}/${culling}.`);
    for (const run of group) {
      validateMeasurement(run.benchmark, configuration);
      if (run.benchmark.rendering.first.culling_enabled !== culling || run.benchmark.rendering.last.culling_enabled !== culling) throw failure("invalid_report", "$.runs", "Observed culling mode differs from its run label.");
    }
    result.push({ workload: workload.id, culling, repeats: group.length, median_of_runs: groupedMetrics(group), repeat_spread: {
      total_cpu_median_ms: statistics(group.map(run => run.benchmark.cpu_ms.total.median)),
      frame_interval_median_ms: statistics(group.map(run => run.benchmark.frame_interval_ms.median)),
    } });
  }
  if (runs.length !== result.length * configuration.repeats) throw failure("invalid_report", "$.runs", "Report includes unexpected workload or culling runs.");
  return result;
}

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])]));
  return value;
}
const same = (a, b) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));

export function validateReport(report) {
  if (report?.version !== REPORT_VERSION || report.kind !== "browser_engine_benchmark" || report.suite !== SUITE || report.status !== "complete") throw failure("invalid_baseline", "$", "Expected a complete report from the current benchmark suite/version.");
  const configuration = report.configuration;
  if (!configuration || !Array.isArray(configuration.workloads) || configuration.workloads.length !== 5 || new Set(configuration.workloads.map(workload => workload.id)).size !== 5 || !Number.isInteger(configuration.repeats) || configuration.repeats < 1 || configuration.repeats > 5 || !Number.isInteger(configuration.warmup) || configuration.warmup < 0 || configuration.warmup > 120 || !Number.isInteger(configuration.samples) || configuration.samples < 1 || configuration.samples > 240 || !["step", "render", "present"].includes(configuration.mode) || !report.environment || !report.source?.scene || !Array.isArray(report.runs) || report.runs.length > 50) throw failure("invalid_baseline", "$", "Report metadata or run limits are invalid.");
  const setup = configuration.setup_ticks ?? 0;
  if (!Number.isInteger(setup) || setup < 0 || setup > 120 || (configuration.mode === "present" ? configuration.presentation_alpha !== 0.5 : configuration.presentation_alpha !== undefined)) throw failure("invalid_baseline", "$.configuration", "Setup ticks or presentation alpha are invalid.");
  return summarizeRuns(report.runs, configuration);
}

export function assertCompatible(current, baseline) {
  if (!same(current.configuration, baseline.configuration)) throw failure("incompatible_baseline", "$.configuration", "Workload, camera, viewport, sample, warm-up, mode or repeat configuration differs from baseline.");
  if (!same(current.environment, baseline.environment)) throw failure("incompatible_baseline", "$.environment", "Browser/version, platform, CPU, adapter probe or launch settings differ from baseline.");
  if (!same(current.source.scene, baseline.source.scene)) throw failure("incompatible_baseline", "$.source.scene", "Authored scene snapshot differs from baseline.");
  const views = report => report.runs.map(run => ({ workload: run.workload, culling: run.culling, repeat: run.repeat, environment: run.benchmark.environment, camera: run.camera })).sort((a, b) => `${a.workload}/${a.culling}/${a.repeat}`.localeCompare(`${b.workload}/${b.culling}/${b.repeat}`));
  if (!same(views(current), views(baseline))) throw failure("incompatible_baseline", "$.runs", "Observed canvas sizes, pixel ratio, user agent or cameras differ from baseline.");
}

export function difference(current, baseline) {
  nonnegative(current, "$.current"); nonnegative(baseline, "$.baseline");
  return { current, baseline, ratio: baseline === 0 ? (current === 0 ? 1 : null) : current / baseline,
    percent: baseline === 0 ? (current === 0 ? 0 : null) : (current / baseline - 1) * 100,
    zero_baseline: baseline === 0 && current > 0 };
}

export const COMPARISON_METRICS = ["total_cpu_median_ms", "total_cpu_p95_ms", "frame_interval_median_ms", "frame_interval_p95_ms", "cpu_asset_bytes", "gpu_mesh_bytes", "gpu_texture_bytes"];

export function compareReports(current, baseline, maximumPercent) {
  const currentRows = validateReport(current), baselineRows = validateReport(baseline);
  assertCompatible(current, baseline);
  const comparisons = currentRows.map(row => {
    const previous = baselineRows.find(other => other.workload === row.workload && other.culling === row.culling);
    return { workload: row.workload, culling: row.culling, metrics: Object.fromEntries(COMPARISON_METRICS.map(metric => [metric, difference(row.median_of_runs[metric], previous.median_of_runs[metric])])) };
  });
  const regressions = [];
  if (maximumPercent !== undefined) for (const row of comparisons) for (const [metric, value] of Object.entries(row.metrics)) {
    if (value.zero_baseline || value.percent > maximumPercent) regressions.push({ workload: row.workload, culling: row.culling, metric, ...value });
  }
  return { compatible: true, max_regression_percent: maximumPercent ?? null, passed: regressions.length === 0, comparisons, regressions };
}

export function compareCulling(summary) {
  return summary.filter(row => row.culling).map(row => {
    const disabled = summary.find(other => other.workload === row.workload && !other.culling);
    return { workload: row.workload, enabled_over_disabled: Object.fromEntries([...COMPARISON_METRICS, "upload_bytes", "visible_instances", "draw_calls"].map(metric => [metric, difference(row.median_of_runs[metric], disabled.median_of_runs[metric])])) };
  });
}
