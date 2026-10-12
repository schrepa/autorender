/** A/B measurement in one development session: variant A's and B's
 * commands (which set state outright, never toggle it) take turns before
 * measurement windows in ABBA order, so drift cancels; each window runs a
 * benchmark, and with `--profile on` also a profiling capture whose GPU
 * spans (and native sessions' per-draw times) it reads, after each variant
 * ran once unmeasured. The report gives
 * each variant's median of window medians, B against A, and the noise the
 * repeated windows of one variant show (A against A, B against B). */
// @ts-check
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { AgentError } from "./agent-protocol.mjs";
import { workspace } from "./workspace.mjs";

const median = values => {
  const sorted = [...values].sort((a, b) => a - b);
  if (!sorted.length) return null;
  const middle = (sorted.length - 1) / 2;
  return (sorted[Math.floor(middle)] + sorted[Math.ceil(middle)]) / 2;
};

/** The commands `text` (JSON) lists: one command or an array of them. */
function commands(text, name) {
  let value;
  try { value = JSON.parse(text ?? "[]"); } catch (error) { throw new AgentError("invalid_arguments", `--${name} is JSON: ${error.message}`, name); }
  const list = Array.isArray(value) ? value : [value];
  if (list.some(command => !command || typeof command !== "object" || typeof command.op !== "string")) {
    throw new AgentError("invalid_arguments", `--${name} lists commands with an op.`, name);
  }
  return list;
}

const integer = (value, fallback, minimum, maximum, name) => {
  if (value === undefined) return fallback;
  const number = Number(value);
  if (!Number.isInteger(number) || number < minimum || number > maximum) throw new AgentError("invalid_arguments", `--${name} is ${minimum} to ${maximum}.`, name);
  return number;
};

/** Runs the A/B `values` describe in `session` (a launched development
 * session): its report. */
export async function runAb(session, values) {
  const variants = { a: commands(values.a, "a"), b: commands(values.b, "b") };
  const rounds = integer(values.rounds, 2, 1, 20, "rounds");
  const samples = integer(values.samples, 60, 1, 600, "samples");
  const warmup = integer(values.warmup, 10, 0, 600, "warmup");
  const mode = values.mode ?? "present";
  if (!["step", "render", "present"].includes(mode)) throw new AgentError("invalid_arguments", "--mode is step, render or present.", "mode");
  const profile = values.profile === "on";
  const durationMs = integer(values["profile-ms"], 2000, 100, 5000, "profile-ms");
  const run = async command => {
    const reply = await session.execute(command);
    if (!reply.ok) throw new AgentError(reply.error?.code ?? "bridge_error", `${command.op}: ${reply.error?.message ?? "failed"}`, reply.error?.path);
    return reply.result;
  };
  const status = await run({ op: "status" });
  if (status.paused !== true) await run({ op: "pause" });
  for (const command of [...variants.a, ...variants.b]) {
    if (["resume", "pause", "benchmark", "profile", "close"].includes(command.op)) throw new AgentError("invalid_arguments", `variants set state; ${command.op} is the harness's own.`);
  }
  // Each variant once unmeasured first: the frames that first draw it
  // compile its pipelines.
  for (const variant of ["a", "b"]) {
    for (const command of variants[variant]) await run(command);
    await run({ op: "benchmark", mode, warmup, samples: Math.min(samples, 10) });
    if (profile) await run({ op: "profile", duration_ms: Math.min(durationMs, 500) });
  }
  const windows = [];
  // ABBA blocks: each variant measured early and late in every block.
  const order = Array.from({ length: rounds }, () => ["a", "b", "b", "a"]).flat();
  for (const [index, variant] of order.entries()) {
    for (const command of variants[variant]) await run(command);
    const benchmark = (await run({ op: "benchmark", mode, warmup, samples })).benchmark;
    const window = {
      index, variant,
      frame_interval_ms: benchmark.frame_interval_ms.median,
      cpu_ms: benchmark.cpu_ms.total.median,
      gpu_ms: {},
      draws_ms: {},
    };
    if (profile) {
      const result = await run({ op: "profile", duration_ms: durationMs });
      const report = JSON.parse(await readFile(resolve(workspace, result.artifacts.trace), "utf8"));
      const gpu = report.gpu ?? report.engine?.gpu ?? {};
      const spans = new Map();
      for (const sample of gpu.samples ?? []) spans.set(sample.name, [...(spans.get(sample.name) ?? []), sample.duration_ms]);
      for (const [name, durations] of spans) window.gpu_ms[name] = median(durations);
      const draws = new Map();
      for (const draw of gpu.draw_samples ?? []) {
        if (draw.duration_ms !== null) draws.set(draw.label, [...(draws.get(draw.label) ?? []), draw.duration_ms]);
      }
      for (const [label, durations] of draws) window.draws_ms[label] = median(durations);
      window.profile = result.artifacts.trace;
    }
    windows.push(window);
  }
  // A measure's median over each variant's windows, B against A, and the
  // spread of each variant's windows against their own median.
  const measures = new Set(["frame_interval_ms", "cpu_ms", ...windows.flatMap(window => Object.keys(window.gpu_ms).map(name => `gpu:${name}`))]);
  const read = (window, measure) => measure.startsWith("gpu:") ? window.gpu_ms[measure.slice(4)] : window[measure];
  const compared = {};
  for (const measure of measures) {
    const of = variant => windows.filter(window => window.variant === variant).map(window => read(window, measure)).filter(value => typeof value === "number");
    const [a, b] = [of("a"), of("b")];
    const [ma, mb] = [median(a), median(b)];
    if (ma === null || mb === null) continue;
    const spread = (values, center) => center ? Math.max(...values.map(value => Math.abs(value - center))) / center : null;
    compared[measure] = {
      a: ma, b: mb, b_minus_a: mb - ma, b_over_a: ma ? mb / ma : null,
      noise: { a: spread(a, ma), b: spread(b, mb) },
    };
  }
  // The draws whose time changed most, by their medians over each variant.
  const labels = new Set(windows.flatMap(window => Object.keys(window.draws_ms)));
  const draws = [...labels].map(label => {
    const of = variant => median(windows.filter(window => window.variant === variant).map(window => window.draws_ms[label]).filter(value => typeof value === "number"));
    const [a, b] = [of("a"), of("b")];
    return { label, a, b, b_minus_a: a !== null && b !== null ? b - a : null };
  }).filter(draw => draw.b_minus_a !== null).sort((x, y) => Math.abs(/** @type {number} */ (y.b_minus_a)) - Math.abs(/** @type {number} */ (x.b_minus_a))).slice(0, 20);
  return {
    order: order.join(""), mode, warmup, samples, profile: profile ? { duration_ms: durationMs } : null,
    variants, compared, draws, windows,
    note: "Noise is each variant's largest window difference from its own median, as a share: B against A means little within it. Frame intervals include the GPU's work waited for after each frame natively, and the browser's pacing in browsers.",
  };
}
