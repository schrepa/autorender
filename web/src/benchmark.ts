/** Bounded sampling for the opt-in development host. No ordinary-frame work. */
import type { RuntimeStats } from "./host";
import { percentiles } from "./statistics.ts";

/** A benchmark control as the engine validated it, with its defaults. */
export type BenchmarkOptions = { mode: "step" | "render" | "present"; warmup: number; samples: number };
export type RenderingStats = RuntimeStats;
export type Observation = { frame: string; tick: string; scene: { id: string; revision: string } | null };
export type Reply = {
  version?: number; ok: boolean; observation?: Observation;
  result?: Record<string, unknown>;
  error?: { code: string; path: string; message: string };
};
export class BenchmarkError extends Error {
  code: string;
  constructor(code: string, message: string) { super(message); this.code = code; }
}

/** The frame statistics in `json`, which agents receive as `rendering`. */
export function readRendering(json: string): RenderingStats {
  return JSON.parse(json) as RenderingStats;
}

/** Fields that are numbers in every one of `measurements`. */
function numericFields(measurements: RenderingStats[]) {
  return Object.keys(measurements[0] ?? {}).filter(field => measurements.every(stats => typeof stats[field] === "number"));
}

/** Percentiles of measured `values`; counts describe actual samples. */
function distribution(values: number[]) {
  if (!values.length || values.some(value => !Number.isFinite(value) || value < 0)) {
    throw new BenchmarkError("invalid_measurement", "measurements must be nonempty, finite and nonnegative");
  }
  return percentiles(values);
}

type Context = {
  signal: AbortSignal;
  invoke(command: Record<string, unknown>): Reply;
  stats(): string;
  size(): [number, number];
  interrupted(): boolean;
};

/** A request owns the host's busy boundary throughout all awaited frames. */
export async function sampleBenchmark(context: Context, options: BenchmarkOptions) {
  const startTime = performance.now();
  const deadline = startTime + 10_000;
  const initialSize = context.size();
  if (initialSize.some(size => size <= 0) || context.interrupted()) {
    throw new BenchmarkError("benchmark_interrupted", "benchmark needs a visible, nonzero paused canvas");
  }
  function check() {
    context.signal.throwIfAborted();
    if (context.interrupted() || context.size().some((value, index) => value !== initialSize[index])) {
      throw new BenchmarkError("benchmark_interrupted", "visibility, canvas size or paused state changed during benchmark");
    }
    if (performance.now() >= deadline) throw new BenchmarkError("benchmark_timeout", "benchmark exceeded ten seconds; reduce samples or warmup; gameplay may have advanced");
  }
  function invoke(command: Record<string, unknown>) {
    check();
    const reply = context.invoke(command);
    if (!reply.ok) throw new BenchmarkError(reply.error?.code ?? "benchmark_failed", reply.error?.message ?? "runtime rejected benchmark frame");
    return reply;
  }
  async function nextFrame(): Promise<number> {
    check();
    return new Promise((resolve, reject) => {
      let frame = 0;
      const timeout = setTimeout(() => finish(new BenchmarkError("benchmark_timeout", "benchmark exceeded ten seconds waiting for browser frame; gameplay may have advanced")), Math.max(1, deadline - performance.now()));
      function finish(error?: unknown, timestamp?: number) {
        clearTimeout(timeout); cancelAnimationFrame(frame);
        context.signal.removeEventListener("abort", abort);
        if (error) reject(error); else resolve(timestamp!);
      }
      function abort() { finish(context.signal.reason ?? new Error("benchmark session closed")); }
      context.signal.addEventListener("abort", abort, { once: true });
      frame = requestAnimationFrame(timestamp => {
        try { check(); finish(undefined, timestamp); } catch (error) { finish(error); }
      });
    });
  }
  const initial = invoke({ op: "assets" });
  invoke({ op: "input" });
  let previousTimestamp = await nextFrame();
  const frames: number[] = [];
  const measurements: RenderingStats[] = [];
  let end: Observation | undefined;
  for (let index = 0; index < options.warmup + options.samples; index++) {
    const reply = invoke(options.mode === "step" ? { op: "step", ticks: 1 }
      : options.mode === "present" ? { op: "present", alpha: 0.5 } : { op: "render" });
    const stats = readRendering(context.stats());
    check();
    // Pair CPU work with the following scheduling interval, including the final
    // sample; the interval preceding this work belongs to the preceding frame.
    const timestamp = await nextFrame();
    if (index >= options.warmup) {
      frames.push(Math.max(0, timestamp - previousTimestamp));
      for (const [field, value] of Object.entries(stats)) {
        if (typeof value === "number" && (!Number.isFinite(value) || value < 0)) throw new BenchmarkError("invalid_measurement", `invalid ${field} statistic`);
      }
      measurements.push(stats);
    }
    previousTimestamp = timestamp;
    end = reply.observation;
  }
  const final = invoke({ op: "assets" });
  const range = (field: string) => ({ min: Math.min(...measurements.map(stats => stats[field] as number)), max: Math.max(...measurements.map(stats => stats[field] as number)) });
  return {
    version: 1, ...options, elapsed_ms: performance.now() - startTime,
    frame_interval_ms: distribution(frames),
    cpu_ms: {
      simulation: distribution(measurements.map(stats => stats.simulation_ms)),
      extraction: distribution(measurements.map(stats => stats.extraction_ms)),
      render: distribution(measurements.map(stats => stats.render_ms)),
      total: distribution(measurements.map(stats => stats.simulation_ms + stats.extraction_ms + stats.render_ms)),
    },
    rendering: { first: measurements[0], last: measurements.at(-1)!, ranges: Object.fromEntries(numericFields(measurements).map(field => [field, range(field)])) },
    observation: { start: initial.observation, end }, assets: { before: initial.result, after: final.result },
    environment: { canvas_width: initialSize[0], canvas_height: initialSize[1], device_pixel_ratio: globalThis.devicePixelRatio ?? 1, user_agent: globalThis.navigator?.userAgent ?? "unknown" },
    timing: "CPU timings exclude JSON/stats collection; frame intervals include host work and browser pacing; GPU execution time is not measured",
  };
}
