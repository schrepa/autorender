/** Optional capture tooling. Imported only when profiling Wasm methods exist. */
import type { Runtime } from "./host";

type JsonObject = Record<string, unknown>;
type CpuEvent = { category: string; name: string; kind: string; start_ms: number; duration_ms: number; value: number | null; depth: number; frame: string | number | null; complete: boolean };
type CpuCapture = { version: number; id: string | number; start_ms: number; end_ms: number; max_events: number; dropped_events: number; incomplete_spans: number; complete: boolean; events: CpuEvent[]; allocations_start: JsonObject; allocations_end: JsonObject; allocations_delta: JsonObject };
type RustCapture = { cpu: CpuCapture; gpu: JsonObject; memory: { wasm_linear_bytes?: number } };
type ProfilingRuntime = Runtime & Required<Pick<Runtime, "profiling_start" | "profiling_stop" | "profiling_status" | "profiling_poll" | "profiling_report">>;
type Frame = { index: number; raf_timestamp_ms: number; start_ms: number; rust_end_ms: number | null; end_ms: number | null; previous_raf_ms: number | null };
type BrowserEntry = { entryType: string; name: string; startTime: number; duration: number; [key: string]: unknown };
type Memory = { timestamp_ms: number; wasm_linear_bytes: number | null; js_heap: { used_bytes: number; total_bytes: number; limit_bytes: number } | null };
type TraceEvent = { name: string; ph: string; pid: number; tid: number; ts?: number; dur?: number; cat?: string; s?: string; args: JsonObject };
type ProfileOptions = { duration_ms?: number; max_events?: number };
export type DetailedProfileReport = {
  version: 1; kind: "stress_detailed_profile"; complete: boolean; reason: string; started_at: string;
  metadata: JsonObject; capabilities: JsonObject; timing: JsonObject; bounds: JsonObject;
  raw_rust: RustCapture | null; browser: { entries: BrowserEntry[]; frames: Frame[]; memory: Memory[]; dropped_entries: number; dropped_frames: number };
  traceEvents: TraceEvent[]; displayTimeUnit: "ms"; summary: JsonObject; errors: string[];
};
type ProfileApi = { version: 1; status(): JsonObject; start(options?: ProfileOptions): JsonObject; stop(): Promise<JsonObject>; export(): DetailedProfileReport };
declare global { interface Window { __engineProfile?: ProfileApi } }

const MAX_FRAMES = 4096, MAX_BROWSER_ENTRIES = 1024, MAX_MEMORY_SAMPLES = 64, MAX_JSON_BYTES = 32 * 1024 * 1024;
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
function json(source: string): JsonObject {
  if (source.length > MAX_JSON_BYTES) throw new Error("Profiling response exceeds 32 MiB");
  const value: unknown = JSON.parse(source);
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Expected a profiling object");
  return value as JsonObject;
}
function copy(value: JsonObject): JsonObject {
  const source = JSON.stringify(value);
  if (source.length > 32 * 1024) throw new Error("Profiling metadata exceeds 32 KiB");
  return JSON.parse(source) as JsonObject;
}
export function supportsProfiling<T extends Runtime>(runtime: T): runtime is T & ProfilingRuntime {
  return [runtime.profiling_start, runtime.profiling_stop, runtime.profiling_status, runtime.profiling_poll, runtime.profiling_report].every(value => typeof value === "function");
}

/** One capture, bounded host buffers, one Rust event serialization at stop. */
export class DetailedProfiler {
  private phase: "idle" | "capturing" | "draining" = "idle";
  private disposed = false;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private timeout: ReturnType<typeof setTimeout> | undefined;
  private observers: PerformanceObserver[] = [];
  private capabilities: JsonObject = {};
  private metadata: JsonObject = {};
  private options = { duration_ms: 5000, max_events: 65536 };
  private frames: Frame[] = [];
  private entries: BrowserEntry[] = [];
  private memories: Memory[] = [];
  private droppedEntries = 0;
  private droppedFrames = 0;
  private started = 0;
  private ended: number | null = null;
  private startedAt = "";
  private lastRaf: number | null = null;
  private current: Frame | undefined;
  private rustStatus: JsonObject = {};
  private report: DetailedProfileReport | undefined;
  private finishing: Promise<JsonObject> | undefined;
  private api: ProfileApi;

  constructor(private readonly runtime: ProfilingRuntime, private readonly context: {
    allowed(): void; metadata(): JsonObject; memory(): WebAssembly.Memory | undefined;
    changed(): void; completed(report: DetailedProfileReport): void;
  }) {
    this.rustStatus = json(runtime.profiling_status());
    this.api = { version: 1, status: () => this.status(), start: options => this.start(options), stop: () => this.stop(), export: () => this.export() };
    window.__engineProfile = this.api;
  }
  get busy(): boolean { return this.phase !== "idle"; }
  get active(): boolean { return this.phase === "capturing"; }
  status(): JsonObject {
    return { available: !this.disposed, active: this.active, busy: this.busy, phase: this.phase, elapsed_ms: this.busy ? performance.now() - this.started : null,
      capture_id: this.report?.raw_rust?.cpu.id ?? this.rustStatus.capture_id ?? null, duration_ms: this.options.duration_ms, max_events: this.options.max_events,
      last_capture: this.report ? { complete: this.report.complete, reason: this.report.reason, summary: this.report.summary } : null,
      backend: this.rustStatus };
  }
  start(options: ProfileOptions = {}): JsonObject {
    if (this.disposed || this.busy) throw new Error("A detailed profile is busy or unavailable");
    const settings = { duration_ms: 5000, max_events: 65536, ...options };
    if (!options || typeof options !== "object" || Array.isArray(options) || Object.keys(options).some(key => !["duration_ms", "max_events"].includes(key)) ||
      !Number.isInteger(settings.duration_ms) || settings.duration_ms < 100 || settings.duration_ms > 5000 ||
      !Number.isInteger(settings.max_events) || settings.max_events < 256 || settings.max_events > 65536) throw new Error("Profile duration must be 100..5000 ms and max_events 256..65536");
    this.context.allowed();
    const metadata = copy(this.context.metadata());
    this.runtime.profiling_start(settings.max_events);
    try { this.rustStatus = json(this.runtime.profiling_status()); }
    catch (error) { this.runtime.profiling_stop(); throw error; }
    this.options = settings; this.metadata = metadata; this.frames = []; this.entries = []; this.memories = [];
    this.droppedEntries = 0; this.droppedFrames = 0; this.lastRaf = null; this.current = undefined;
    this.startedAt = new Date().toISOString(); this.started = performance.now(); this.ended = null;
    this.phase = "capturing"; this.finishing = undefined; this.report = undefined;
    this.capabilities = { performance_observer: typeof PerformanceObserver !== "undefined", supported_entry_types: typeof PerformanceObserver === "undefined" ? [] : [...PerformanceObserver.supportedEntryTypes], observed_entry_types: [], observer_errors: [], performance_time_origin_ms: performance.timeOrigin,
      user_agent: navigator.userAgent, hardware_concurrency: navigator.hardwareConcurrency, cross_origin_isolated: crossOriginIsolated, js_heap: "memory" in performance, wasm_linear_memory: !!this.context.memory(), gc_events: "unavailable: no portable GC event API", gpu: this.rustStatus.gpu ?? null };
    for (const type of ["longtask", "long-animation-frame"]) {
      if (typeof PerformanceObserver === "undefined" || !PerformanceObserver.supportedEntryTypes.includes(type)) continue;
      try {
        const observer = new PerformanceObserver(list => this.receive(list.getEntries()));
        observer.observe({ type, buffered: false }); this.observers.push(observer);
        (this.capabilities.observed_entry_types as string[]).push(type);
      } catch (error) { (this.capabilities.observer_errors as string[]).push(String(error).slice(0, 256)); }
    }
    this.sampleMemory(); this.context.changed();
    this.timer = setTimeout(() => { void this.finish(true, "duration_complete"); }, settings.duration_ms);
    this.timeout = setTimeout(() => { void this.finish(false, "deadline_exceeded"); }, 15_000);
    return this.status();
  }
  beforeFrame(timestamp: number): void {
    if (!this.active) return;
    if (performance.now() - this.started >= 15_000) { void this.finish(false, "deadline_exceeded"); return; }
    if (this.frames.length === MAX_FRAMES) { this.droppedFrames++; void this.finish(false, "host_frame_limit"); return; }
    this.current = { index: this.frames.length, raf_timestamp_ms: timestamp, start_ms: performance.now(), rust_end_ms: null, end_ms: null, previous_raf_ms: this.lastRaf };
    this.frames.push(this.current); this.lastRaf = timestamp;
  }
  afterFrame(): void { if (this.active && this.current) this.current.rust_end_ms = performance.now(); }
  afterHostFrame(): void {
    if (!this.active || !this.current) return;
    this.current.end_ms = performance.now(); this.current = undefined;
    if (this.memories.length < MAX_MEMORY_SAMPLES && performance.now() - (this.memories.at(-1)?.timestamp_ms ?? 0) >= 250) this.sampleMemory();
  }
  stop(): Promise<JsonObject> { return this.finish(false, "stopped_by_user"); }
  abort(reason: string): void { if (this.busy) void this.finish(false, reason); }
  export(): DetailedProfileReport {
    if (this.busy || !this.report) throw new Error("Stop the detailed profile before exporting");
    return this.report;
  }
  dispose(): void {
    this.abort("host_reload"); this.disposed = true;
    if (window.__engineProfile === this.api) delete window.__engineProfile;
  }
  private receive(entries: PerformanceEntry[]): void {
    for (const entry of entries) {
      if (entry.startTime + entry.duration < this.started || (this.ended !== null && entry.startTime > this.ended)) continue;
      if (this.entries.length === MAX_BROWSER_ENTRIES) { this.droppedEntries++; continue; }
      const source = entry as PerformanceEntry & Record<string, unknown>;
      const value: BrowserEntry = { entryType: entry.entryType, name: entry.name.slice(0, 256), startTime: entry.startTime, duration: entry.duration };
      for (const key of ["blockingDuration", "renderStart", "styleAndLayoutStart", "firstUIEventTimestamp"]) if (finite(source[key])) value[key] = source[key];
      // Script/attribution lists can grow independently of the number of entries.
      for (const field of ["scripts", "attribution"]) if (Array.isArray(source[field])) {
        value[`${field}_total`] = source[field].length;
        value[field] = source[field].slice(0, 16).map(item => {
          const compact: JsonObject = {};
          for (const key of ["name", "entryType", "containerType", "containerName", "containerSrc", "invoker", "invokerType", "sourceURL", "sourceFunctionName", "startTime", "duration", "executionStart", "forcedStyleAndLayoutDuration", "pauseDuration", "sourceCharPosition"]) {
            const part: unknown = (item as Record<string, unknown>)[key];
            if (finite(part)) compact[key] = part; else if (typeof part === "string") compact[key] = part.slice(0, 256);
          }
          return compact;
        });
      }
      // Keep unusual attribution strings/lists from dominating a bounded export.
      for (const field of ["scripts", "attribution"]) {
        const parts = value[field];
        if (Array.isArray(parts)) while (parts.length && JSON.stringify(value).length > 2048) parts.pop();
      }
      this.entries.push(value);
    }
  }
  private sampleMemory(): void {
    const value = (performance as Performance & { memory?: { usedJSHeapSize: number; totalJSHeapSize: number; jsHeapSizeLimit: number } }).memory;
    const heap = value && [value.usedJSHeapSize, value.totalJSHeapSize, value.jsHeapSizeLimit].every(finite) ? { used_bytes: value.usedJSHeapSize, total_bytes: value.totalJSHeapSize, limit_bytes: value.jsHeapSizeLimit } : null;
    this.memories.push({ timestamp_ms: performance.now(), wasm_linear_bytes: this.context.memory()?.buffer.byteLength ?? null, js_heap: heap });
  }
  private finish(complete: boolean, reason: string): Promise<JsonObject> {
    if (this.finishing) return this.finishing;
    if (!this.busy) return Promise.resolve(this.status());
    this.ended = performance.now(); this.phase = "draining";
    clearTimeout(this.timer); clearTimeout(this.timeout); this.context.changed();
    const elapsed = this.ended - this.started;
    if (elapsed >= 15_000) { complete = false; reason = "deadline_exceeded"; }
    this.finishing = this.finalize(complete, reason).catch(error => {
      for (const observer of this.observers) observer.disconnect();
      this.observers = []; this.phase = "idle";
      const failure: DetailedProfileReport = {
        version: 1, kind: "stress_detailed_profile", complete: false, reason: "capture_failed", started_at: this.startedAt,
        metadata: this.metadata, capabilities: this.capabilities, timing: { browser_start_ms: this.started, browser_end_ms: this.ended },
        bounds: this.options, raw_rust: null,
        browser: { entries: this.entries, frames: this.frames, memory: this.memories, dropped_entries: this.droppedEntries, dropped_frames: this.droppedFrames },
        traceEvents: [], displayTimeUnit: "ms", summary: { cpu_events: 0, dropped_cpu_events: 0, host_frames: this.frames.length, browser_entries: this.entries.length, gpu: {} },
        errors: [String(error).slice(0, 1000)],
      };
      this.report = failure;
      try { this.context.changed(); this.context.completed(failure); } catch { /* Retain export even if the UI was torn down. */ }
      return this.status();
    });
    return this.finishing;
  }
  private async finalize(complete: boolean, reason: string): Promise<JsonObject> {
    const errors: string[] = []; let raw: RustCapture | null = null;
    try {
      const result = json(this.runtime.profiling_stop()) as unknown as RustCapture;
      if (!result.cpu || !Array.isArray(result.cpu.events) || result.cpu.events.length > 65536) throw new Error("Invalid bounded CPU capture");
      raw = result;
      const deadline = Math.min(this.started + 15_000, performance.now() + 250);
      while (!this.disposed && performance.now() < deadline) {
        this.runtime.profiling_poll();
        this.rustStatus = json(this.runtime.profiling_status());
        const gpu = this.rustStatus.gpu as JsonObject | undefined;
        if (!gpu || gpu.pending === 0 || gpu.supported === false) break;
        await new Promise(resolve => setTimeout(resolve, 16));
      }
      if (!this.disposed) { this.runtime.profiling_poll(); raw.gpu = json(this.runtime.profiling_report()); this.rustStatus = json(this.runtime.profiling_status()); }
    } catch (error) { errors.push(String(error).slice(0, 1000)); complete = false; reason = "capture_failed"; }
    for (const observer of this.observers) { this.receive(observer.takeRecords()); observer.disconnect(); }
    this.observers = [];
    if (this.memories.length < MAX_MEMORY_SAMPLES) this.sampleMemory();
    if (this.disposed) { complete = false; reason = "host_reload"; }
    if (performance.now() - this.started >= 15_000) { complete = false; reason = "deadline_exceeded"; }
    if (raw && (!raw.cpu.complete || raw.cpu.dropped_events > 0 || raw.cpu.incomplete_spans > 0)) { complete = false; if (reason === "duration_complete") reason = "cpu_capture_incomplete"; }
    if (raw?.gpu.supported === true && raw.gpu.complete !== true) { complete = false; if (reason === "duration_complete") reason = "gpu_capture_incomplete"; }
    if (this.droppedEntries || this.droppedFrames) { complete = false; if (reason === "duration_complete") reason = "host_buffer_limit"; }
    const { samples: _gpuSamples, ...gpuSummary } = raw?.gpu ?? {};
    const summary = { cpu_events: raw?.cpu.events.length ?? 0, dropped_cpu_events: raw?.cpu.dropped_events ?? 0, host_frames: this.frames.length, browser_entries: this.entries.length,
      gpu: gpuSummary, allocations: raw?.cpu.allocations_delta ?? null, memory_start: this.memories[0] ?? null, memory_end: this.memories.at(-1) ?? null };
    const result: DetailedProfileReport = {
      version: 1, kind: "stress_detailed_profile", complete, reason, started_at: this.startedAt, metadata: this.metadata, capabilities: this.capabilities,
      timing: { browser_time_origin_ms: performance.timeOrigin, browser_start_ms: this.started, browser_end_ms: this.ended,
        cpu_clock: "Rust and browser spans use window performance.now milliseconds; trace timestamps are microseconds from that same origin", gpu_clock: "GPU elapsed durations remain in raw_rust.gpu. GPU timestamps have no calibrated CPU correlation. Trace GPU duration counters are positioned at their associated CPU frame solely for attribution; they are not GPU start times or execution spans. Samples without a matching CPU frame remain only in raw data.",
        overhead: "Profiling scopes, allocator atomics, GPU queries/readback, bounded JS frame records, four-Hz memory reads, browser observers and the ordinary HUD add overhead. CPU/Rust buffers serialize once after stop; trace conversion/export occurs after capture. Host spans include Wasm and HUD; nested spans must not be summed. No per-entity instrumentation.",
        browser_limits: "Observers expose only browser-supported entries; their absence does not prove the absence of stalls. Entries overlapping the capture retain their full reported durations. Script attributions are capped at 16 per entry and trimmed to a 2048-character entry budget; total counts show truncation. Timers cannot preempt a blocked browser thread: the 15-second deadline is enforced at the next yield. JS heap is an optional browser estimate, Wasm linear memory is reserved capacity, allocator counters and GPU payload are distinct scopes. No inferred GC events or total process memory." },
      bounds: { ...this.options, deadline_ms: 15_000, gpu_drain_ms: 250, host_frames: MAX_FRAMES, browser_entries: MAX_BROWSER_ENTRIES, memory_samples: MAX_MEMORY_SAMPLES, export_bytes: MAX_JSON_BYTES },
      raw_rust: raw, browser: { entries: this.entries, frames: this.frames, memory: this.memories, dropped_entries: this.droppedEntries, dropped_frames: this.droppedFrames },
      traceEvents: trace(raw, this.frames, this.entries, this.memories), displayTimeUnit: "ms", summary, errors,
    };
    this.report = result; this.phase = "idle"; this.context.changed(); this.context.completed(result);
    return this.status();
  }
}

function trace(raw: RustCapture | null, frames: Frame[], entries: BrowserEntry[], memories: Memory[]): TraceEvent[] {
  const result: TraceEvent[] = [];
  const cpuFrames = new Map<string, number>();
  for (const [tid, name] of [[1, "Rust CPU scopes"], [2, "Host RAF callbacks (includes Rust)"], [3, "Browser long tasks"], [4, "Browser long animation frames"], [5, "Memory snapshots"]] as const) result.push({ name: "thread_name", ph: "M", pid: 1, tid, args: { name } });
  for (const event of raw?.cpu.events ?? []) {
    if (!finite(event.start_ms) || !finite(event.duration_ms)) continue;
    if (event.frame !== null && event.name === "browser.frame") {
      const frame = String(event.frame), previous = cpuFrames.get(frame);
      if (previous === undefined || event.start_ms < previous) cpuFrames.set(frame, event.start_ms);
    }
    const common = { name: event.name, cat: event.category, pid: 1, tid: 1, ts: event.start_ms * 1000 };
    if (event.kind === "counter") result.push({ ...common, ph: "C", args: { value: event.value ?? 0 } });
    else if (event.kind === "aggregate") result.push({ ...common, ph: "C", args: { summed_duration_ms: event.duration_ms, count: event.value ?? 0 } });
    else result.push({ ...common, ph: "X", dur: Math.max(0, event.duration_ms) * 1000, args: { frame: event.frame, depth: event.depth, complete: event.complete } });
  }
  for (const frame of frames) {
    if (frame.end_ms !== null) result.push({ name: "RAF host callback", cat: "browser.host", ph: "X", pid: 1, tid: 2, ts: frame.start_ms * 1000, dur: (frame.end_ms - frame.start_ms) * 1000, args: { host_frame: frame.index, raf_timestamp_ms: frame.raf_timestamp_ms } });
    if (frame.rust_end_ms !== null) result.push({ name: "Wasm frame and input consumption", cat: "browser.wasm", ph: "X", pid: 1, tid: 2, ts: frame.start_ms * 1000, dur: (frame.rust_end_ms - frame.start_ms) * 1000, args: { host_frame: frame.index } });
    if (frame.previous_raf_ms !== null) result.push({ name: "RAF interval ms", cat: "browser.scheduling", ph: "C", pid: 1, tid: 2, ts: frame.raf_timestamp_ms * 1000, args: { interval_ms: frame.raf_timestamp_ms - frame.previous_raf_ms } });
  }
  for (const entry of entries) result.push({ name: entry.entryType, cat: "browser.observer", ph: "X", pid: 1, tid: entry.entryType === "longtask" ? 3 : 4, ts: entry.startTime * 1000, dur: entry.duration * 1000, args: entry });
  for (const memory of memories) {
    const values: JsonObject = {};
    if (memory.wasm_linear_bytes !== null) values.wasm_linear_bytes = memory.wasm_linear_bytes;
    if (memory.js_heap) values.js_heap_used_bytes = memory.js_heap.used_bytes;
    result.push({ name: "Memory bytes (separate scopes)", cat: "memory", ph: "C", pid: 1, tid: 5, ts: memory.timestamp_ms * 1000, args: values });
  }
  if (raw && Array.isArray(raw.gpu.samples)) {
    result.push({ name: "process_name", ph: "M", pid: 2, tid: 1, args: { name: "GPU duration counters — CPU frame attribution only" } });
    result.push({ name: "thread_name", ph: "M", pid: 2, tid: 1, args: { name: "GPU elapsed ms (not an aligned GPU timeline)" } });
    for (const source of raw.gpu.samples.slice(0, 4096)) {
      const sample = source as Record<string, unknown>, timestamp = cpuFrames.get(String(sample.frame));
      if (timestamp !== undefined && finite(sample.duration_ms)) result.push({ name: String(sample.name ?? "GPU pass"), cat: "gpu.duration_by_cpu_frame", ph: "C", pid: 2, tid: 1, ts: timestamp * 1000, args: { duration_ms: sample.duration_ms } });
    }
  }
  return result;
}
