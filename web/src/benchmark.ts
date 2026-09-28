/** Bounded sampling for the opt-in development host. No ordinary-frame work. */
import type { RuntimeStats } from "./host";

export type BenchmarkOptions = { mode: "step" | "render" | "present"; warmup: number; samples: number };
const flagFields = ["culling_enabled", "shadows_enabled", "ambient_occlusion", "bloom", "sun_shafts", "reflections", "refraction"] as const;
// Read back a few frames late: null until then.
const casterFields = ["shadow_near_casters", "shadow_moving_casters", "shadow_kept_casters", "lights_visible", "light_cluster_overflow", "occluded_instances", "subpixel_instances"] as const;
export type RenderingStats = Omit<RuntimeStats, "free" | typeof casterFields[number]> & Record<typeof casterFields[number], number | null>;
export type Observation = { frame: string; tick: string; scene: { id: string; revision: string } | null };
export type Reply = { ok: boolean; observation?: Observation; result?: Record<string, unknown>; error?: { code: string; path: string; message: string } };
const numericFields = [
  "render_candidates", "visible_instances", "culled_instances", "instances", "dropped_instances", "missing_assets", "invalid_transforms",
  "draw_calls", "upload_bytes", "fixed_steps", "dropped_seconds", "resident_meshes", "resident_mesh_bytes", "mesh_upload_bytes",
  "resident_textures", "resident_texture_bytes", "texture_upload_bytes", "simulation_ms", "extraction_ms", "render_ms",
  "shadow_draws", "shadow_kept_redraw", "local_lights", "dropped_lights", "transparent_draws", "particles",
  "dropped_particles", "decals", "dropped_decals", "reflection_probes", "reflection_probe_faces",
  "dropped_reflection_probes", "virtual_clusters", "virtual_triangles", "virtual_shadow_clusters",
  "virtual_shadow_triangles", "virtual_occluded_clusters", "dropped_virtual",
] as const;

export class BenchmarkError extends Error {
  code: string;
  constructor(code: string, message: string) { super(message); this.code = code; }
}

export function benchmarkOptions(command: Record<string, unknown>): BenchmarkOptions {
  if (Object.keys(command).some(key => !["op", "mode", "warmup", "samples"].includes(key))) {
    throw new BenchmarkError("invalid_benchmark", "benchmark accepts only mode, warmup and samples");
  }
  const mode = command.mode ?? "step";
  const warmup = command.warmup ?? 30;
  const samples = command.samples ?? 120;
  if ((mode !== "step" && mode !== "render" && mode !== "present") || !Number.isInteger(warmup) || !Number.isInteger(samples) ||
      typeof warmup !== "number" || warmup < 0 || warmup > 120 || typeof samples !== "number" || samples < 1 || samples > 240) {
    throw new BenchmarkError("invalid_benchmark", "mode must be step, render or present; warmup 0..120; samples 1..240");
  }
  return { mode, warmup, samples };
}

/** Copy aggregate fields and release the generated Wasm binding immediately. */
export function readRendering(stats: RuntimeStats): RenderingStats {
  try {
    const result = {} as RenderingStats;
    for (const field of flagFields) result[field] = stats[field];
    for (const field of numericFields) result[field] = stats[field];
    for (const field of casterFields) result[field] = stats[field] ?? null;
    return result;
  } finally { stats.free(); }
}

/** Linear interpolation of sorted observations; counts describe actual samples. */
export function distribution(values: number[]) {
  if (!values.length || values.some(value => !Number.isFinite(value) || value < 0)) {
    throw new BenchmarkError("invalid_measurement", "measurements must be nonempty, finite and nonnegative");
  }
  const sorted = [...values].sort((a, b) => a - b);
  function quantile(fraction: number) {
    const index = (sorted.length - 1) * fraction;
    const lower = Math.floor(index);
    return sorted[lower] + (sorted[Math.ceil(index)] - sorted[lower]) * (index - lower);
  }
  return { min: sorted[0], median: quantile(0.5), p95: quantile(0.95), p99: quantile(0.99), max: sorted.at(-1)! };
}

type Context = {
  signal: AbortSignal;
  invoke(command: Record<string, unknown>): Reply;
  stats(): RuntimeStats;
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
      for (const field of numericFields) {
        if (!Number.isFinite(stats[field]) || stats[field] < 0) throw new BenchmarkError("invalid_measurement", `invalid ${field} statistic`);
      }
      measurements.push(stats);
    }
    previousTimestamp = timestamp;
    end = reply.observation;
  }
  const final = invoke({ op: "assets" });
  const range = (field: typeof numericFields[number]) => ({ min: Math.min(...measurements.map(stats => stats[field])), max: Math.max(...measurements.map(stats => stats[field])) });
  return {
    version: 1, ...options, elapsed_ms: performance.now() - startTime,
    frame_interval_ms: distribution(frames),
    cpu_ms: {
      simulation: distribution(measurements.map(stats => stats.simulation_ms)),
      extraction: distribution(measurements.map(stats => stats.extraction_ms)),
      render: distribution(measurements.map(stats => stats.render_ms)),
      total: distribution(measurements.map(stats => stats.simulation_ms + stats.extraction_ms + stats.render_ms)),
    },
    rendering: { first: measurements[0], last: measurements.at(-1)!, ranges: Object.fromEntries(numericFields.map(field => [field, range(field)])) },
    observation: { start: initial.observation, end }, assets: { before: initial.result, after: final.result },
    environment: { canvas_width: initialSize[0], canvas_height: initialSize[1], device_pixel_ratio: globalThis.devicePixelRatio ?? 1, user_agent: globalThis.navigator?.userAgent ?? "unknown" },
    timing: "CPU timings exclude JSON/stats collection; frame intervals include host work and browser pacing; GPU execution time is not measured",
  };
}
