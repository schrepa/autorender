/** Production opt-in recording. The host owns frames, lifecycle and all UI.
 * Call abort when configuration, camera, resolution, visibility or pause changes.
 * sample copies primitive values; callers still own/free any Wasm stats binding.
 */
import { type Distribution, percentiles } from "./statistics.ts";

type StressJson = null | boolean | number | string | StressJson[] | { [key: string]: StressJson };
type StressRecordingMetadata = Record<string, unknown>;
type StressRecordingOptions = {
  warmup?: number; duration_ms?: number; timeout_ms?: number; max_samples?: number;
};
type StressFrameInput = {
  timestamp: number;
  simulation_ms: number; extraction_ms: number; render_ms: number;
  dropped_seconds: number;
  /** Additional phase timers may be subsets of simulation/extraction/render. */
  phases_ms?: Record<string, number>;
  /** Per-frame aggregate counters; retained as first/last/ranges, not summed. */
  counters?: Record<string, number | boolean>;
};
type Options = Required<StressRecordingOptions>;
type Frame = {
  elapsed_ms: number; frame_interval_ms: number;
  simulation_ms: number; extraction_ms: number; render_ms: number; total_cpu_ms: number;
  phases_ms: Record<string, number>;
};
type Counters = Record<string, number | boolean>;
type Pending = StressFrameInput & { phases_ms: Record<string, number>; counters: Counters };
type Diagnostic = { elapsed_ms: number; measured: boolean; snapshot: StressJson };
type Range = { min: number; max: number };
export type StressRecordingReport = {
  version: 1; kind: "stress_recording"; complete: boolean; reason: string;
  started_at: string; metadata: StressJson; environment: Record<string, StressJson>;
  options: Options; warmup_frames: number; samples: number;
  elapsed_ms: number; measured_ms: number;
  frame_interval_ms: Distribution | null;
  cpu_ms: { simulation: Distribution | null; extraction: Distribution | null; render: Distribution | null; total: Distribution | null };
  phase_subsets_ms: Record<string, Distribution>;
  rendering: { first: Counters | null; last: Counters | null; ranges: Record<string, Range> };
  dropped_seconds: { start: number | null; end: number | null; delta: number | null };
  diagnostics: Diagnostic[]; frames: Frame[]; timing: string;
};

const DEFAULTS: Options = { warmup: 30, duration_ms: 10_000, timeout_ms: 30_000, max_samples: 1800 };
const MAX_DIAGNOSTICS = 121;

function jsonCopy(value: unknown, maxBytes: number): StressJson {
  const source = JSON.stringify(value, (_key, item: unknown) => {
    if (typeof item === "number" && !Number.isFinite(item)) throw new Error("recording metadata must contain finite numbers");
    if (typeof item === "bigint" || typeof item === "function" || typeof item === "symbol" || item === undefined) {
      throw new Error("recording metadata must be JSON data");
    }
    return item;
  });
  if (!source || new TextEncoder().encode(source).length > maxBytes) throw new Error("recording JSON exceeds its byte limit");
  return JSON.parse(source) as StressJson;
}

function checkedOptions(options: StressRecordingOptions): Options {
  const result = { ...DEFAULTS, ...options };
  if (Object.keys(options).some(key => !Object.hasOwn(DEFAULTS, key)) || Object.values(result).some(value => !Number.isInteger(value)) ||
      result.warmup < 0 || result.warmup > 120 || result.duration_ms < 1 || result.duration_ms > 10_000 ||
      result.timeout_ms < result.duration_ms || result.timeout_ms > 30_000 || result.max_samples < 1 || result.max_samples > 1800) {
    throw new Error("recording requires warmup0..120, duration1..10000ms, timeout duration..30000ms and max_samples1..1800");
  }
  return result;
}

function finite(value: number): boolean { return Number.isFinite(value) && value >= 0; }
function values<T extends number | boolean>(input: Record<string, T> | undefined, maximum: number): Record<string, T> {
  const entries = Object.entries(input ?? {});
  if (entries.length > maximum || entries.some(([key, value]) => key.length > 64 || !/^[A-Za-z][A-Za-z0-9_]*$/.test(key) ||
      (typeof value !== "boolean" && (typeof value !== "number" || !finite(value))))) {
    throw new Error("recording fields must be bounded named finite nonnegative numbers or booleans");
  }
  return Object.fromEntries(entries);
}

/** The development benchmark's percentiles, or null without samples. */
function distribution(input: number[]): Distribution | null {
  return input.length ? percentiles(input) : null;
}

/** No adapter request/device allocation: pass the runtime's existing adapter info
 * in start metadata if available. Browser hints do not prove physical GPU use.
 */
function environment(): Record<string, StressJson> {
  const browser = globalThis.navigator;
  const memory = (browser as Navigator & { deviceMemory?: number } | undefined)?.deviceMemory;
  return {
    user_agent: browser?.userAgent ?? "unknown", platform: browser?.platform ?? "unknown",
    language: browser?.language ?? "unknown", hardware_concurrency: browser?.hardwareConcurrency ?? null,
    device_memory_gib: memory ?? null, webgpu_available: !!browser && "gpu" in browser,
    device_pixel_ratio: globalThis.devicePixelRatio ?? 1,
    viewport_width: globalThis.innerWidth ?? null, viewport_height: globalThis.innerHeight ?? null,
    adapter: null,
  };
}

/** A single bounded recording. Incomplete reports remain exportable, clearly
 * marked with the interruption reason. Starting while active throws.
 */
export class StressRecorder {
  private running = false;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private options: Options = DEFAULTS;
  private metadata: StressJson = null;
  private browser: Record<string, StressJson> = {};
  private startedAt = "";
  private started = 0;
  private firstTimestamp: number | undefined;
  private previous: Pending | undefined;
  private processed = 0;
  private measured = 0;
  private frames: Frame[] = [];
  private diagnostics: Diagnostic[] = [];
  private lastDiagnostic = -Infinity;
  private firstCounters: Counters | null = null;
  private lastCounters: Counters | null = null;
  private ranges: Record<string, Range> = {};
  private dropStart: number | null = null;
  private dropEnd: number | null = null;
  private previousDropped: number | undefined;
  private counterKeys: string | undefined;
  private phaseKeys: string | undefined;

  constructor(private readonly onComplete: (report: StressRecordingReport) => void) {}
  get active(): boolean { return this.running; }

  start(metadata: StressRecordingMetadata, options: StressRecordingOptions = {}): void {
    if (this.running) throw new Error("a stress recording is already active");
    // Validate before mutating the previous recording state.
    const configured = checkedOptions(options);
    const copied = jsonCopy(metadata, 16 * 1024);
    this.options = configured; this.metadata = copied; this.browser = environment();
    this.startedAt = new Date().toISOString(); this.started = performance.now();
    this.firstTimestamp = undefined; this.previous = undefined;
    this.processed = 0; this.measured = 0; this.frames = []; this.diagnostics = [];
    this.lastDiagnostic = -Infinity; this.firstCounters = null; this.lastCounters = null; this.ranges = {};
    this.dropStart = null; this.dropEnd = null; this.previousDropped = undefined;
    this.counterKeys = undefined; this.phaseKeys = undefined;
    this.running = true;
    this.timer = setTimeout(() => this.finish(false, "timeout"), configured.timeout_ms);
  }

  sample(input: StressFrameInput): void {
    if (!this.running) return;
    if (performance.now() - this.started >= this.options.timeout_ms) { this.finish(false, "timeout"); return; }
    try {
      if (![input.timestamp, input.simulation_ms, input.extraction_ms, input.render_ms, input.dropped_seconds].every(finite) ||
          !finite(input.simulation_ms + input.extraction_ms + input.render_ms) ||
          (this.previous && input.timestamp <= this.previous.timestamp) ||
          (this.previous && input.dropped_seconds < this.previous.dropped_seconds)) {
        throw new Error("invalid or nonmonotonic frame statistics");
      }
      const phases = values(input.phases_ms, 16);
      if (Object.values(phases).some(value => typeof value !== "number")) throw new Error("phase timers must be numbers");
      const counters = values(input.counters, 40);
      const counterKeys = Object.keys(counters).sort().join(",");
      const phaseKeys = Object.keys(phases).sort().join(",");
      if ((this.counterKeys !== undefined && counterKeys !== this.counterKeys) ||
          (this.phaseKeys !== undefined && phaseKeys !== this.phaseKeys)) throw new Error("recording statistic fields changed");
      this.counterKeys = counterKeys; this.phaseKeys = phaseKeys;
      this.firstTimestamp ??= input.timestamp;
      if (this.previous) {
        const previous = this.previous;
        if (this.processed >= this.options.warmup) {
          const interval = input.timestamp - previous.timestamp;
          this.frames.push({
            elapsed_ms: previous.timestamp - this.firstTimestamp, frame_interval_ms: interval,
            simulation_ms: previous.simulation_ms, extraction_ms: previous.extraction_ms, render_ms: previous.render_ms,
            total_cpu_ms: previous.simulation_ms + previous.extraction_ms + previous.render_ms, phases_ms: previous.phases_ms,
          });
          this.measured += interval;
          this.firstCounters ??= previous.counters; this.lastCounters = previous.counters;
          for (const [key, value] of Object.entries(previous.counters)) {
            const numeric = typeof value === "boolean" ? Number(value) : value;
            const range = this.ranges[key];
            this.ranges[key] = { min: range ? Math.min(range.min, numeric) : numeric, max: range ? Math.max(range.max, numeric) : numeric };
          }
          this.dropStart ??= this.previousDropped ?? previous.dropped_seconds;
          this.dropEnd = previous.dropped_seconds;
        }
        this.previousDropped = previous.dropped_seconds;
        this.processed++;
      }
      // The final measured CPU sample belongs to the interval just completed,
      // not the new frame whose stats arrived with this callback.
      if (this.measured >= this.options.duration_ms) { this.finish(true, "duration_complete"); return; }
      if (this.frames.length >= this.options.max_samples) { this.finish(false, "sample_limit"); return; }
      this.previous = { ...input, phases_ms: phases, counters };
    } catch (error) {
      this.finish(false, `invalid_measurement: ${String(error).slice(0, 180)}`);
    }
  }

  /** At most four JSON snapshots per second, including warm-up. */
  diagnostic(timestamp: number, snapshot: unknown): void {
    if (!this.running) return;
    if (!finite(timestamp)) { this.abort("invalid_diagnostic_timestamp"); return; }
    if (timestamp - this.lastDiagnostic < 250 || this.diagnostics.length >= MAX_DIAGNOSTICS) return;
    try {
      this.diagnostics.push({ elapsed_ms: Math.max(0, timestamp - (this.firstTimestamp ?? timestamp)),
        measured: this.processed >= this.options.warmup, snapshot: jsonCopy(snapshot, 8 * 1024) });
      this.lastDiagnostic = timestamp;
    } catch { this.abort("invalid_diagnostic_snapshot"); }
  }

  abort(reason = "cancelled"): void {
    if (this.running) this.finish(false, reason.slice(0, 200));
  }

  private finish(complete: boolean, reason: string): void {
    if (!this.running) return;
    this.running = false;
    clearTimeout(this.timer); this.timer = undefined; this.previous = undefined;
    const phaseNames = Object.keys(this.frames[0]?.phases_ms ?? {});
    const report: StressRecordingReport = {
      version: 1, kind: "stress_recording", complete, reason,
      started_at: this.startedAt, metadata: this.metadata, environment: this.browser, options: this.options,
      warmup_frames: Math.min(this.processed, this.options.warmup), samples: this.frames.length,
      elapsed_ms: performance.now() - this.started, measured_ms: this.measured,
      frame_interval_ms: distribution(this.frames.map(frame => frame.frame_interval_ms)),
      cpu_ms: {
        simulation: distribution(this.frames.map(frame => frame.simulation_ms)),
        extraction: distribution(this.frames.map(frame => frame.extraction_ms)),
        render: distribution(this.frames.map(frame => frame.render_ms)),
        total: distribution(this.frames.map(frame => frame.total_cpu_ms)),
      },
      phase_subsets_ms: Object.fromEntries(phaseNames.map(name => [name, distribution(this.frames.map(frame => frame.phases_ms[name]))!])),
      rendering: { first: this.firstCounters, last: this.lastCounters, ranges: this.ranges },
      dropped_seconds: { start: this.dropStart, end: this.dropEnd,
        delta: this.dropStart === null || this.dropEnd === null ? null : this.dropEnd - this.dropStart },
      diagnostics: this.diagnostics, frames: this.frames,
      timing: "CPU total is simulation + extraction + render preparation/submission. Additional phase timers are overlapping subsets, not additive. CPU timers exclude recording/stat collection; RAF intervals include host work and browser pacing. GPU execution time is not measured. Adapter/browser hints do not establish physical GPU use. Dropped time is the change in the observed cumulative counter; without warm-up its first frame has no preceding baseline.",
    };
    this.onComplete(report);
  }
}
