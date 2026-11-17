import type { Runtime } from "./host";
import { adapterInfo } from "./adapter";
import { fatal } from "./panic";
import { engineError, engineFailure, type EngineError } from "../generated/errors";
import { READ_BACK_LIMIT_MS, type AdapterStatus } from "../generated/host";
import type { Effect, Error as Failure, HostContext, Outcome, Reply, Step } from "../generated/reply";
import { parseJson } from "../generated/runtime";
import protocol from "../generated/protocol.json";

type Command = { verb: string; [field: string]: unknown };
interface Host {
  game: Runtime;
  signal: AbortSignal;
  paused(): boolean;
  setPaused(value: boolean): void;
  exclusive<R>(work: () => Promise<R>): Promise<R> | undefined;
  size(): [number, number];
  poll(): void;
  frame(timestamp: number): void;
  gpuWorkDone?(): Promise<void> | undefined;
  fail(error: unknown): void;
  stopped(): string | undefined;
  source?: { url: string; load(signal: AbortSignal): Promise<string>; files(source: string, signal: AbortSignal): Promise<unknown> };
  controls?: Record<string, PageControl>;
  parameters?: Record<string, string>;
}

export interface PageControl {
  values: string;
  run(args: Record<string, unknown>): unknown;
}

declare global {
  interface Window {
    __engineDev?: {
      version: number; instance: string; controls: Record<string, string>; parameters: Record<string, string>;
      request(command: Command): Promise<Reply>;
      saved(path: string): string | undefined;
      waitingOnGpu(): boolean;
    };
  }
}

export function connectDevelopment(host: Host): () => void {
  let disposed = false;
  const runtime = host.game;
  if (!runtime.dev_request) throw engineFailure({ code: "runtime_unavailable", path: "$", expected: "present", got: "dev_request" });
  const closed = () => disposed || host.signal.aborted;
  let gpuWaits = 0;
  const saved = new Map<string, string>();
  async function save(name: string, contents: unknown): Promise<Outcome> {
    const text = JSON.stringify(contents);
    const bytes = new TextEncoder().encode(text);
    const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
    const path = `page/${name}-${saved.size + 1}.json`;
    saved.set(path, text);
    return { artifact: { path, mime_type: "application/json", bytes: bytes.byteLength, sha256: Array.from(digest, byte => byte.toString(16).padStart(2, "0")).join("") } };
  }
  async function presented(count: number): Promise<void> {
    const done = host.gpuWorkDone?.();
    if (!done) {
      await animationFrames(count, host.signal);
      return;
    }
    gpuWaits++;
    const gpu = within(done, READ_BACK_LIMIT_MS, "gpu_work", host.signal)
      .finally(() => { gpuWaits--; });
    await animationFrames(count, host.signal, gpu);
  }
  function context(): string {
    const [width, height] = host.size();
    const page: HostContext = { width, height, paused: host.paused(), source: host.source?.url ?? null };
    return JSON.stringify(page);
  }
  function refuse(failure: Failure): Reply {
    if (!closed() && !host.stopped()) {
      try { return parseJson(runtime.dev_refusal!(JSON.stringify(failure))); } catch {}
    }
    return { ok: false, error: failure };
  }
  async function carry(effect: Effect): Promise<Step> {
    const resumed = (outcome: Outcome): Step => {
      if (closed()) throw engineFailure(CLOSED);
      return parseJson(runtime.dev_resume!(JSON.stringify(outcome), context()));
    };
    switch (effect.kind) {
      case "pause":
        host.setPaused(effect.paused);
        return resumed({});
      case "frame":
        await presented(1);
        if (closed()) throw engineFailure(CLOSED);
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
        const files = await host.source.files(source, host.signal);
        if (closed()) throw engineFailure(CLOSED);
        return parseJson(runtime.dev_reloaded!(source, context(), files));
      }
      case "save":
        return resumed(await save(effect.name, effect.contents));
    }
  }
  const api = {
    version: protocol.version,
    instance: crypto.randomUUID(),
    controls: Object.fromEntries(Object.entries(host.controls ?? {}).map(([name, control]) => [name, control.values])),
    parameters: host.parameters ?? {},
    waitingOnGpu: () => gpuWaits > 0,
    saved(path: string): string | undefined {
      const text = saved.get(path);
      saved.delete(path);
      return text;
    },
    async request(command: Command): Promise<Reply> {
      if (closed()) return refuse(CLOSED);
      const stopped = host.stopped();
      if (stopped) return { ok: false, error: { code: "runtime_panicked", path: "$", got: stopped } };
      const answered = host.exclusive(async (): Promise<Reply> => {
        let line: string;
        try { line = JSON.stringify(command) ?? "null"; }
        catch { return refuse({ code: "invalid_request", path: "$.command", expected: "json" }); }
        try {
          let step = parseJson(runtime.dev_request!(line, context()));
          while ("effect" in step) step = await carry(step.effect);
          const reply = step.reply;
          if (closed()) return refuse(CLOSED);
          if (reply.ok) {
            await nameAdapters(reply);
            host.poll();
          }
          return reply;
        } catch (cause) {
          if (closed()) return refuse(CLOSED);
          if (fatal(cause)) {
            host.fail(cause);
            return { ok: false, error: { code: "runtime_panicked", path: "$", got: host.stopped() ?? String(cause) } };
          }
          const thrown = engineError(cause);
          if (thrown) return refuse(failure(thrown));
          return refuse({ code: "browser_command_failed", path: "$", got: String(cause) });
        }
      });
      return answered ?? { ok: false, error: { code: "command_busy", path: "$" } };
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

function failure(thrown: EngineError): Failure {
  const { code, path, expected, allowed, nearest, got } = thrown;
  return { code, path, expected, allowed, nearest, got };
}

async function nameAdapters(reply: Reply) {
  const result = reply.result as { adapter?: AdapterStatus } | null | undefined;
  if (!result?.adapter) return;
  const browser = await adapterInfo();
  result.adapter = { ...result.adapter, vendor: result.adapter.vendor || browser?.vendor || "", architecture: browser?.architecture ?? "" };
}

const PRESENTATION_TIMEOUT_MS = 5000;
const CLOSED: Failure = { code: "session_closed", path: "$" };

function within<T>(promise: Promise<T>, limit: number, waited: string, signal: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => finish(() => reject(engineFailure({ code: "command_timeout", path: "$", got: { waited, limit_ms: limit } }))), limit);
    function finish(settle: () => void) {
      clearTimeout(timeout);
      signal.removeEventListener("abort", aborted);
      settle();
    }
    function aborted() { finish(() => reject(engineFailure(CLOSED))); }
    signal.addEventListener("abort", aborted, { once: true });
    if (signal.aborted) { aborted(); return; }
    promise.then(value => finish(() => resolve(value)), cause => finish(() => reject(cause)));
  });
}

function animationFrames(count: number, signal: AbortSignal, gpu?: Promise<void>): Promise<void> {
  return new Promise((resolve, reject) => {
    let request = 0;
    let left = count;
    let settled = false;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    const arm = () => {
      if (settled) return;
      timeout = setTimeout(() => finish(engineFailure({ code: "command_timeout", path: "$", got: { waited: "presentation", limit_ms: PRESENTATION_TIMEOUT_MS } })), PRESENTATION_TIMEOUT_MS);
    };
    function finish(error?: Error) {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      cancelAnimationFrame(request);
      signal.removeEventListener("abort", aborted);
      if (error) reject(error); else resolve();
    }
    function aborted() { finish(engineFailure(CLOSED)); }
    function next() {
      if (--left <= 0) finish();
      else request = requestAnimationFrame(next);
    }
    signal.addEventListener("abort", aborted, { once: true });
    if (signal.aborted) { aborted(); return; }
    if (gpu) gpu.then(arm, (cause: unknown) => finish(cause instanceof Error ? cause : new Error(String(cause))));
    else arm();
    request = requestAnimationFrame(next);
  });
}
