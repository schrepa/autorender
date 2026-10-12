/** On-demand adapter, dynamically imported only by the Vite development
 * host: the engine's development bridge runs each request and the control
 * it names step by step; the page carries out what only it can (`Effect`)
 * and resumes the request with how it went. */
import type { Runtime } from "./host";
import { adapterInfo } from "./adapter";
import { fatal } from "./panic";
import { engineError, type ErrorCode } from "../generated/errors";
import type { AdapterStatus } from "../generated/host";
import type { Effect, HostContext, Outcome, Reply, Step } from "../generated/reply";
import { parseJson } from "../generated/runtime";
// The protocol's version, generated from the engine (`cargo xtask schema`).
import protocol from "../generated/protocol.json";

type Command = { op: string; [field: string]: unknown };
interface Host {
  game: Runtime;
  signal: AbortSignal;
  paused(): boolean;
  setPaused(value: boolean): void;
  setBusy(value: boolean): void;
  isBusy?(): boolean;
  size(): [number, number];
  poll(): void;
  /** Runs one ordinary frame at `timestamp` without input, then hands its
   * jobs to the workers. */
  frame(timestamp: number): void;
  /** Resolves once the GPU has done the work submitted so far, pipeline
   * compiles among it; undefined without a WebGPU canvas. */
  gpuWorkDone?(): Promise<void> | undefined;
  /** Stops the page for good: the engine's module stopped. */
  fail(error: unknown): void;
  /** Why the engine's module stopped (its panic's message), once it did. */
  stopped(): string | undefined;
  source?: { url: string; load(signal: AbortSignal): Promise<string>; files?(source: string, signal: AbortSignal): Promise<unknown> };
  /** The game's own controls, which the `game` control runs by name. */
  controls?: Record<string, PageControl>;
  /** The page parameters the host and game read, with what each does. */
  parameters?: Record<string, string>;
}

/** A control a game's page registers: what it does, and how. */
export interface PageControl {
  description: string;
  run(args: Record<string, unknown>): unknown;
}

declare global {
  interface Window {
    __engineDev?: {
      version: number; instance: string; build: string | null; controls: Record<string, string>; parameters: Record<string, string>;
      request(command: Command): Promise<Reply>;
      /** Whether a request waits on the GPU before presenting: browsers
       * present nothing while their GPU process compiles pipelines or
       * catches up with queued frames, for tens of seconds at times. */
      waitingOnGpu(): boolean;
    };
  }
}

/** What a benchmark's frame intervals include on a page. */
const TIMING = "CPU timings exclude JSON/stats collection; frame intervals include host work and browser pacing; GPU execution time is not measured";

/** Requests run between frames; no borrowed Wasm views escape. */
export function connectDevelopment(host: Host): () => void {
  let disposed = false;
  let busy = false;
  const runtime = host.game;
  if (!runtime.dev_request) throw new Error("Development Wasm is missing; run cargo xtask dev.");
  const closed = () => disposed || host.signal.aborted;
  /** Presentation waits on the GPU in progress. */
  let gpuWaits = 0;
  /** Resolves within the `count`th animation frame from now, as
   * animationFrames: its timeout runs once the GPU has done the work
   * submitted so far. */
  async function presented(count: number): Promise<void> {
    const done = host.gpuWorkDone?.();
    if (!done) {
      await animationFrames(count, host.signal);
      return;
    }
    gpuWaits++;
    const gpu = within(done, GPU_WORK_LIMIT_MS, "the GPU did not finish its work", host.signal)
      .finally(() => { gpuWaits--; });
    await animationFrames(count, host.signal, undefined, gpu);
  }
  /** The page's side of each call to the bridge. */
  function context(): string {
    const [width, height] = host.size();
    const page: HostContext = {
      width, height, paused: host.paused(), device_pixel_ratio: globalThis.devicePixelRatio ?? 1,
      user_agent: globalThis.navigator?.userAgent ?? "unknown", source: host.source?.url ?? null, timing: TIMING,
    };
    return JSON.stringify(page);
  }
  /** A refusal the page decided, with the engine's observation while it runs. */
  function refuse(code: ErrorCode, path: string, message: string): Reply {
    if (!closed() && !host.stopped()) {
      try { return parseJson(runtime.dev_refusal!(code, path, message)); } catch { /* Preserve the original failure. */ }
    }
    return { ok: false, error: { code, path, message } };
  }
  /** Carries out `effect` for the request under way: its next step. */
  async function carry(effect: Effect): Promise<Step> {
    const resumed = (outcome: Outcome): Step => {
      if (closed()) throw new Error("session closed");
      return parseJson(runtime.dev_resume!(JSON.stringify(outcome), context()));
    };
    switch (effect.kind) {
      case "pause":
        host.setPaused(effect.paused);
        return resumed({});
      case "wait": {
        const now = await animationFrames(1, host.signal, effect.within_ms);
        return resumed(globalThis.document?.hidden ? { interrupted: true } : { now });
      }
      case "frame":
        await presented(1);
        if (closed()) throw new Error("session closed");
        host.frame(effect.timestamp);
        return resumed({});
      case "show":
        await presented(2);
        return resumed({});
      case "game": {
        const control = host.controls?.[effect.name];
        if (!control) return resumed({ missing: Object.keys(host.controls ?? {}) });
        let outcome: Outcome;
        try { outcome = { result: await control.run(effect.args) ?? null }; }
        catch (cause) { outcome = { failed: String(cause) }; }
        return resumed(outcome);
      }
      case "source": {
        if (!host.source) return resumed({ missing: [] });
        const source = await host.source.load(host.signal);
        const files = await host.source.files?.(source, host.signal);
        if (closed()) throw new Error("session closed");
        return parseJson(runtime.dev_reloaded!(source, context(), files));
      }
    }
  }
  const api = {
    version: protocol.version,
    instance: crypto.randomUUID(),
    build: new URLSearchParams(globalThis.location?.search ?? "").get("engine_build"),
    /** The game's controls by name, with what each does. */
    controls: Object.fromEntries(Object.entries(host.controls ?? {}).map(([name, control]) => [name, control.description])),
    parameters: host.parameters ?? {},
    waitingOnGpu: () => gpuWaits > 0,
    async request(command: Command): Promise<Reply> {
      if (closed()) return refuse("session_closed", "$", "the browser runtime was disposed");
      const stopped = host.stopped();
      if (stopped) return { ok: false, error: { code: "runtime_panicked", path: "$", message: stopped } };
      if (busy || host.isBusy?.()) return refuse("command_busy", "$", "one development command is already running");
      let line: string;
      try { line = JSON.stringify(command) ?? "null"; }
      catch { return refuse("invalid_request", "$.command", "command must be serializable JSON"); }
      busy = true;
      host.setBusy(true);
      try {
        let step = parseJson(runtime.dev_request!(line, context()));
        while ("effect" in step) step = await carry(step.effect);
        const reply = step.reply;
        if (closed()) return refuse("session_closed", "$", "runtime was disposed during command");
        if (reply.ok) {
          await nameAdapters(reply);
          host.poll();
        }
        return reply;
      } catch (cause) {
        if (closed()) return refuse("session_closed", "$", "runtime was disposed during command");
        if (fatal(cause)) {
          host.fail(cause);
          return { ok: false, error: { code: "runtime_panicked", path: "$", message: host.stopped() ?? String(cause) } };
        }
        // The engine's own errors keep their code and path.
        const thrown = engineError(cause);
        if (thrown) return refuse(thrown.code, thrown.path, thrown.message);
        return refuse("browser_command_failed", "$", String(cause));
      } finally {
        busy = false;
        host.setBusy(false);
      }
    },
  };
  window.__engineDev = api;
  host.signal.addEventListener("abort", dispose, { once: true });
  function dispose() {
    disposed = true;
    if (window.__engineDev === api) delete window.__engineDev;
  }
  return dispose;
}

/** The GPUs `reply` names (status's, a benchmark's environment's) with the
 * vendor and architecture the browser names, which the engine cannot read
 * on a page. */
async function nameAdapters(reply: Reply) {
  const result = reply.result as { adapter?: AdapterStatus; benchmark?: { environment?: { adapter?: AdapterStatus } } } | null | undefined;
  for (const holder of [result, result?.benchmark?.environment]) {
    if (!holder?.adapter) continue;
    const browser = await adapterInfo();
    holder.adapter = { ...holder.adapter, vendor: holder.adapter.vendor || browser?.vendor || "", architecture: browser?.architecture ?? "" };
  }
}

/** Milliseconds a presentation may take once the GPU has done the work
 * submitted before it. */
const PRESENTATION_TIMEOUT_MS = 5000;
/** Milliseconds the GPU may take over the work submitted before a
 * presentation: pipeline compiles and queued frames, which take tens of
 * seconds on a busy integrated GPU. */
const GPU_WORK_LIMIT_MS = 180_000;

/** `promise`, failing with `message` after `limit` milliseconds or once
 * `signal` aborts. */
function within<T>(promise: Promise<T>, limit: number, message: string, signal: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => finish(() => reject(new Error(`${message} within ${limit / 1000} s`))), limit);
    function finish(settle: () => void) {
      clearTimeout(timeout);
      signal.removeEventListener("abort", aborted);
      settle();
    }
    function aborted() { finish(() => reject(new Error("session closed"))); }
    signal.addEventListener("abort", aborted, { once: true });
    if (signal.aborted) { aborted(); return; }
    promise.then(value => finish(() => resolve(value)), cause => finish(() => reject(cause)));
  });
}

/** The timestamp of the `count`th animation frame from now (two show what
 * was drawn before); none within `within` milliseconds resolves with the
 * time then, and without `within` none within PRESENTATION_TIMEOUT_MS
 * rejects, a time that starts once `gpu` resolved (now without it). */
function animationFrames(count: number, signal: AbortSignal, within?: number, gpu?: Promise<void>): Promise<number> {
  return new Promise((resolve, reject) => {
    let request = 0;
    let left = count;
    let settled = false;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    // Once the GPU has done the work submitted before the frames: browsers
    // present nothing while their GPU process compiles pipelines or
    // catches up with queued frames.
    const arm = () => {
      if (settled) return;
      timeout = within === undefined
        ? setTimeout(() => finish(new Error("browser presentation timed out")), PRESENTATION_TIMEOUT_MS)
        : setTimeout(() => finish(undefined, performance.now()), within);
    };
    function finish(error?: Error, timestamp = 0) {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      cancelAnimationFrame(request);
      signal.removeEventListener("abort", aborted);
      if (error) reject(error); else resolve(timestamp);
    }
    function aborted() { finish(new Error("session closed")); }
    function next(timestamp: number) {
      if (--left <= 0) finish(undefined, timestamp);
      else request = requestAnimationFrame(next);
    }
    signal.addEventListener("abort", aborted, { once: true });
    if (signal.aborted) { aborted(); return; }
    if (gpu) gpu.then(arm, (cause: unknown) => finish(cause instanceof Error ? cause : new Error(String(cause))));
    else arm();
    request = requestAnimationFrame(next);
  });
}
