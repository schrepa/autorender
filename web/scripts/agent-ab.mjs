// @ts-check
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { AgentError, detailOf, range, variants as oneOf } from "./agent-protocol.mjs";
import { workspace } from "./workspace.mjs";

function commands(text, name) {
  let value;
  try { value = JSON.parse(text ?? "[]"); } catch { throw new AgentError("invalid_arguments", `--${name}`, { expected: "json" }); }
  const list = Array.isArray(value) ? value : [value];
  if (list.some(command => !command || typeof command !== "object" || typeof command.verb !== "string")) {
    throw new AgentError("invalid_arguments", `--${name}`, { expected: "object" });
  }
  return list;
}

const integer = (value, fallback, minimum, maximum, name) => {
  if (value === undefined) return fallback;
  const number = Number(value);
  if (!Number.isInteger(number) || number < minimum || number > maximum) throw new AgentError("invalid_arguments", `--${name}`, { allowed: range(minimum, maximum), got: value });
  return number;
};

export async function runAb(session, values) {
  const variants = { a: commands(values.a, "a"), b: commands(values.b, "b") };
  const rounds = integer(values.rounds, 2, 1, 20, "rounds");
  const samples = integer(values.samples, 60, 1, 600, "samples");
  const warmup = integer(values.warmup, 10, 0, 600, "warmup");
  const mode = values.mode ?? "present";
  if (!["step", "render", "present", "paced"].includes(mode)) throw new AgentError("invalid_arguments", "--mode", { allowed: oneOf(["step", "render", "present", "paced"]), got: mode });
  const profile = values.profile === "on";
  const durationMs = integer(values["profile-ms"], 2000, 100, 5000, "profile-ms");
  const run = async command => {
    const reply = await session.execute(command);
    if (!reply.ok) throw new AgentError(reply.error?.code ?? "bridge_error", reply.error?.path, detailOf(reply.error));
    return reply.result;
  };
  const control = (name, args = {}) => run({ verb: "control", name, args });
  const median = async values => (values.length ? (await run({ verb: "query", name: "distribution", args: { values } })).distribution?.median ?? null : null);
  const status = await run({ verb: "query", name: "status" });
  if (status.paused !== true) await control("pause");
  for (const command of [...variants.a, ...variants.b]) {
    const owned = command.verb === "profile" || (command.verb === "control" && ["resume", "pause", "benchmark", "close"].includes(command.name));
    if (owned) throw new AgentError("invalid_arguments", "--a", { got: command });
  }
  for (const variant of ["a", "b"]) {
    for (const command of variants[variant]) await run(command);
    await control("benchmark", { mode, warmup, samples: Math.min(samples, 10) });
    if (profile) await run({ verb: "profile", deep: { duration_ms: Math.min(durationMs, 500) } });
  }
  const windows = [];
  const order = Array.from({ length: rounds }, () => ["a", "b", "b", "a"]).flat();
  for (const [index, variant] of order.entries()) {
    for (const command of variants[variant]) await run(command);
    const benchmark = (await control("benchmark", { mode, warmup, samples })).benchmark;
    const window = {
      index, variant,
      frame_interval_ms: benchmark.frame_interval_ms.median,
      cpu_ms: benchmark.cpu_ms.total.median,
      gpu_ms: {},
      draws_ms: {},
    };
    if (profile) {
      const result = await run({ verb: "profile", deep: { duration_ms: durationMs } });
      const report = JSON.parse(await readFile(resolve(workspace, result.artifacts.trace), "utf8"));
      const gpu = report.gpu ?? report.engine?.gpu ?? {};
      const spans = new Map();
      for (const sample of gpu.samples ?? []) spans.set(sample.name, [...(spans.get(sample.name) ?? []), sample.duration_ms]);
      for (const [name, durations] of spans) window.gpu_ms[name] = await median(durations);
      const draws = new Map();
      for (const draw of gpu.draw_samples ?? []) {
        if (draw.duration_ms !== null) draws.set(draw.label, [...(draws.get(draw.label) ?? []), draw.duration_ms]);
      }
      for (const [label, durations] of draws) window.draws_ms[label] = await median(durations);
      window.profile = result.artifacts.trace;
    }
    windows.push(window);
  }
  const measures = new Set(["frame_interval_ms", "cpu_ms", ...windows.flatMap(window => Object.keys(window.gpu_ms).map(name => `gpu:${name}`))]);
  const read = (window, measure) => measure.startsWith("gpu:") ? window.gpu_ms[measure.slice(4)] : window[measure];
  const compared = {};
  for (const measure of measures) {
    const of = variant => windows.filter(window => window.variant === variant).map(window => read(window, measure)).filter(value => typeof value === "number");
    const [a, b] = [of("a"), of("b")];
    const [ma, mb] = [await median(a), await median(b)];
    if (ma === null || mb === null) continue;
    const spread = (values, center) => center ? Math.max(...values.map(value => Math.abs(value - center))) / center : null;
    compared[measure] = {
      a: ma, b: mb, b_minus_a: mb - ma, b_over_a: ma ? mb / ma : null,
      noise: { a: spread(a, ma), b: spread(b, mb) },
    };
  }
  const labels = new Set(windows.flatMap(window => Object.keys(window.draws_ms)));
  const changed = [];
  for (const label of labels) {
    const of = variant => median(windows.filter(window => window.variant === variant).map(window => window.draws_ms[label]).filter(value => typeof value === "number"));
    const [a, b] = [await of("a"), await of("b")];
    if (a !== null && b !== null) changed.push({ label, a, b, b_minus_a: b - a });
  }
  const draws = changed.sort((x, y) => Math.abs(y.b_minus_a) - Math.abs(x.b_minus_a)).slice(0, 20);
  return {
    order: order.join(""), mode, warmup, samples, profile: profile ? { duration_ms: durationMs } : null,
    variants, compared, draws, windows,
  };
}
